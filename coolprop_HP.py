import math
import numpy as np
import sympy
from sympy import Symbol, Function
import torch
from torch import nn
import physicsnemo.sym
from scipy.integrate import solve_ivp

# PhysicsNeMo imports
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.domain import Domain
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.node import Node
import CoolProp.CoolProp as CP
from scipy.interpolate import RegularGridInterpolator


# -----------------------------------
#Cool prop look up tables (interpolated from scipy tables)
# ------------------------------------
p_range = np.linspace(6e6, 10e6, 50)   # pressure, 6-10 MPa for now
h_range = np.linspace(2e5, 5e5, 50)    # 200-500 kJ/kg for now

# making tables
P_grid, H_grid = np.meshgrid(p_range, h_range, indexing='ij')
T_table = np.zeros_like(P_grid)
rho_table = np.zeros_like(P_grid)

for i in range(len(p_range)):
  for j in range(len(h_range)):
    try:
      T_table[i, j] = CP.PropsSI('T',   'P', p_range[i], 'H', h_range[j], 'CO2') # extracts from CP, should be 312 ish at inlet?
      rho_table[i,j] = CP.PropsSI('D',   'P', p_range[i], 'H', h_range[j], 'CO2')
    except:
      T_table[i, j]  = 312.59  # inlet derived vaklues as fallback in case of failure
      rho_table[i, j] = 284.13

# extract properties
T_interp   = RegularGridInterpolator((p_range, h_range), T_table,   method='linear', bounds_error=False, fill_value=None)
rho_interp = RegularGridInterpolator((p_range, h_range), rho_table, method='linear', bounds_error=False, fill_value=None)

# -----------------------------------
# CO2 Heat Pump PDE Definition 
# -----------------------------------
class CO2GasCooler1D(PDE):
    def __init__(self, m_dot, U, P, D, f):
        x = Symbol("x")
        input_variables = {"x": x}

        # Unknown fields
        p = Function("p")(*input_variables)
        h = Function("h")(*input_variables)

        # scaling p and h
        p_scale = 1e7
        h_scale = 1e5

        p_phys = p * p_scale  # physical parameters
        h_phys = h * h_scale 

        # Derivatives
        dp_dx = p.diff(x) * p_scale
        dh_dx = h.diff(x) * h_scale

        # interpolators for custom node
        self.T_interp   = T_interp
        self.rho_interp = rho_interp
        self.m_dot = m_dot
        self.U = U
        self.P = P
        self.D = D
        self.f = f

        # Use symbols, easier for parametric also
        T_sym = Symbol("T_co2")
        rho_sym = Symbol("rho_co2")
        u_sym = m_dot / rho_sym

        # Wall temperature (constant for now - can change maybe make higer?)
        T_wall = 300

        # ---- Equations ----
        self.equations = {}

        # Energy balance
        self.equations["energy"] = (
            (m_dot * dh_dx - U * P * (T_wall - T_sym)) / 1e5
        )

        # Momentum balance - removed for now 
        self.equations["momentum"] = (
            dp_dx + f * rho_sym * u_sym**2 / (2 * D) 
        ) / 1e7 # do we need this scaled?


# co2 derviator 
class CO2PropertyEvaluator(nn.Module):
    def __init__(self, T_interp, rho_interp, p_scale=1e7, h_scale=1e5):
        super().__init__()
        self.T_interp   = T_interp
        self.rho_interp = rho_interp
        self.p_scale    = p_scale
        self.h_scale    = h_scale

    def forward(self, inputs):
        p_phys = inputs["p"].detach().cpu().numpy() * self.p_scale
        h_phys = inputs["h"].detach().cpu().numpy() * self.h_scale

        pts = np.column_stack([p_phys.flatten(), h_phys.flatten()])

        T_vals   = self.T_interp(pts).reshape(p_phys.shape)
        rho_vals = self.rho_interp(pts).reshape(p_phys.shape)

        return {
            "T_co2":   torch.tensor(T_vals,   dtype=inputs["p"].dtype,
                                    device=inputs["p"].device),
            "rho_co2": torch.tensor(rho_vals, dtype=inputs["p"].dtype,
                                    device=inputs["p"].device)
        }

def make_co2_property_node(T_interp, rho_interp):
    evaluator = CO2PropertyEvaluator(T_interp, rho_interp)
    return Node(
        inputs=["p", "h"],
        outputs=["T_co2", "rho_co2"],
        evaluate=evaluator,
        name="co2_property_node"
    )

# -----------------------------------
# 2. Main Run Function
# -----------------------------------
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # -----------------------
    # Physical parameters
    # -----------------------
    m_dot = 0.1         # kg/s
    U = 100.0           # W/m²K
    P = 0.05            # perimeter (m)
    D = 0.01            # diameter (m)
    f = 0.02            # friction factor

    # PDE
    pde = CO2GasCooler1D(m_dot, U, P, D, f)

    x = Symbol("x")

    # -----------------------
    # Neural Network
    # -----------------------
    net = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("p"), Key("h")],
        cfg=cfg.arch.fully_connected
    )

    # original way, changed to follow PN node pattern
   ## property_node = CO2PropertyNode(T_interp, rho_interp) # lookup node

    property_node = make_co2_property_node(T_interp, rho_interp)
    nodes = (pde.make_nodes() + [net.make_node(name="pinn_network")] + [property_node])

    # -----------------------
    # Geometry
    # -----------------------
    #L_gc = 1.0          # gas cooler length
    line = Line1D(0.0, 1.0)

    domain = Domain()

    # -----------------------
    # Boundary Conditions
    # -----------------------

    # Inlet conditions
    p_in = 8e6       # Pa
    h_in = 4e5       # J/kg
  

    bc_inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"p": p_in / 1e7, "h": h_in / 1e5}, #added scaling
        criteria=sympy.Eq(x, 0.0),
        batch_size=cfg.batch_size.bc_inlet
    )
    domain.add_constraint(bc_inlet, "bc_inlet")

    # Outlet pressure (typical design variable later)
    p_out = 7e6

    bc_outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"p": p_out / 1e7}, # added scaling
        criteria=sympy.Eq(x, 1.0), # change to L_gc if reinstating
        batch_size=cfg.batch_size.bc_outlet
    )
    domain.add_constraint(bc_outlet, "bc_outlet") # outlet commented out for now

    # -----------------------
    # Interior Constraints (PDE)
    # -----------------------
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=line,
        outvar={
            "energy": 0,
            "momentum": 0 
        },
        batch_size=cfg.batch_size.interior
    )
    domain.add_constraint(interior, "interior")

    # -----------------------
    # Analytical solution / validator
    # -----------------------

    T_wall = 300.0 # (same as above)
    def co2_odes(x, y):
        p, h = y
        pts = np.array([[p, h]])
        T   = float(T_interp(pts))
        rho = float(rho_interp(pts))
        u   = m_dot / (rho * np.pi * D**2 / 4)
    
        dh_dx = (U * P * (T_wall - T)) / m_dot
        dp_dx = -f * rho * u**2 / (2 * D)
        return [dp_dx, dh_dx]
  
    sol = solve_ivp(
        co2_odes,
        [0, 1.0],
        [p_in, h_in],
        method='RK45',
        dense_output=True,
        max_step=0.005
    )
    
    x_vals   = np.linspace(0, 1, 200).reshape(-1, 1)
    ph_vals  = sol.sol(x_vals.flatten())
    
    p_numerical = ph_vals[0].reshape(-1, 1)
    h_numerical = ph_vals[1].reshape(-1, 1)
    
    validator = PointwiseValidator(
        nodes=nodes,
        invar={"x": x_vals},
        true_outvar={
            "h": h_numerical / 1e5,
            "p": p_numerical / 1e7,
        },
    )
    domain.add_validator(validator, "coolprop_validator")


    # -----------------------
    # Solver
    # -----------------------
    slv = Solver(cfg, domain)
    slv.solve()


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    run()
