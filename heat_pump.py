import math
import numpy as np
import sympy
from sympy import Symbol, Function
import torch
import physicsnemo.sym

# PhysicsNeMo imports
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.domain import Domain
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.eq.pde import PDE


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

        rho = 800  # constant approx (TEMPORARY)
        u = m_dot / rho

        # Wall temperature (constant for now)
        T_wall = 300

        # Simplified T_fluid for now (needs EOS)
        cp = 1000.0  # J/kgK
        T_fluid = h_phys / cp

        # ---- Equations ----
        self.equations = {}

        # Energy balance
        self.equations["energy"] = (
            (m_dot * dh_dx - U * P * (T_wall - T_fluid)) / 1e5
        )

        # Momentum balance - removed for now 
        self.equations["momentum"] = (
            dp_dx + f * rho * u**2 / (2 * D)
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

    nodes = pde.make_nodes() + [net.make_node(name="pinn_network")]

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
            #"momentum": 0 # temporary removed
        },
        batch_size=cfg.batch_size.interior
    )
    domain.add_constraint(interior, "interior")

    # -----------------------
    # Analytical solution / validator
    # -----------------------
    
    x_vals = np.linspace(0, 1, 200).reshape(-1,1)
    # Parameters (same as above)
    m_dot = 0.1         # kg/s
    U = 100.0           # W/m²K
    P = 0.05            # perimeter (m)
    D = 0.01            # diameter (m)
    f = 0.02            # friction factor
    rho = 800.0
    cp = 1000.0
    T_wall = 300.0
    h_in = 4e5
    p_in = 8e6

    # Derived enthalpy analytical solution (for this instant)
    NTU = (U * P) / (m_dot * cp)  # decay constant
    h_analytical = cp * T_wall + (h_in - cp * T_wall) * np.exp(-NTU * x_vals)

    # Pressure analytical solution (drop off. derived also)
    u_vel = m_dot / rho
    dp_dx = -f * rho * u_vel**2 / (2 * D)
    p_analytical = p_in + dp_dx * x_vals

    from physicsnemo.sym.domain.validator import PointwiseValidator

    validator = PointwiseValidator(
        nodes=nodes,
        invar={"x": x_vals},
        true_outvar={
            "h": h_analytical / 1e5,   # scaled to match network output
            "p": p_analytical / 1e7,
        },
    )
    domain.add_validator(validator, "analytical_validator")

    # Don't need to add more values because it is parametric?

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
