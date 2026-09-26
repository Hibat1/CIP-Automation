

import sys
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Tuple

matplotlib.rcParams.update({
    'figure.facecolor': '#0a0c10',
    'axes.facecolor':   '#0f1318',
    'axes.edgecolor':   '#2a3547',
    'axes.labelcolor':  '#8a9ab0',
    'xtick.color':      '#5a6a80',
    'ytick.color':      '#5a6a80',
    'text.color':       '#c8d8e8',
    'grid.color':       '#1a2535',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'lines.linewidth':  1.6,
    'font.family':      'monospace',
    'font.size':        9,
})


class CIPState(IntEnum):
    IDLE          = 0
    PRECHECK      = 1
    INIT_RINSE    = 2
    ALK_WASH      = 3
    INTER_RINSE   = 4
    ACID_WASH     = 5
    FINAL_RINSE   = 6
    COMPLETE      = 7
    HOLDING       = 10
    HELD          = 11
    RESTARTING    = 12
    STOPPING      = 20
    STOPPED       = 21
    ABORTING      = 30
    ABORTED       = 31
    ESTOP         = 99


STATE_NAMES = {
    CIPState.IDLE:        'IDLE',
    CIPState.PRECHECK:    'PRECHECK',
    CIPState.INIT_RINSE:  'INIT RINSE',
    CIPState.ALK_WASH:    'ALK WASH',
    CIPState.INTER_RINSE: 'INTER RINSE',
    CIPState.ACID_WASH:   'ACID WASH',
    CIPState.FINAL_RINSE: 'FINAL RINSE',
    CIPState.COMPLETE:    'COMPLETE',
    CIPState.HOLDING:     'HOLDING',
    CIPState.HELD:        'HELD',
    CIPState.RESTARTING:  'RESTARTING',
    CIPState.STOPPING:    'STOPPING',
    CIPState.STOPPED:     'STOPPED',
    CIPState.ABORTING:    'ABORTING',
    CIPState.ABORTED:     'ABORTED',
    CIPState.ESTOP:       'E-STOP',
}

STATE_COLORS = {
    CIPState.IDLE:        '#5a6a80',
    CIPState.PRECHECK:    '#4de8ff',
    CIPState.INIT_RINSE:  '#69ff9a',
    CIPState.ALK_WASH:    '#ffc845',
    CIPState.INTER_RINSE: '#69ff9a',
    CIPState.ACID_WASH:   '#ff6b6b',
    CIPState.FINAL_RINSE: '#4de8ff',
    CIPState.COMPLETE:    '#00e676',
    CIPState.HOLDING:     '#ffc845',
    CIPState.HELD:        '#ffc845',
    CIPState.ABORTING:    '#ff3b3b',
    CIPState.ABORTED:     '#ff3b3b',
    CIPState.ESTOP:       '#ff0000',
}


@dataclass
class Recipe:
    name: str
    alk_temp_sp: float      # NaOH phase temperature setpoint [°C]
    acid_temp_sp: float     # AcAc phase temperature setpoint [°C]
    dur_init_rinse: float   # [seconds]
    dur_alk_wash: float     # [seconds]
    dur_inter_rinse: float  # [seconds]
    dur_acid_wash: float    # [seconds]
    dur_final_rinse: float  # [seconds]
    cond_gate: float        # Conductivity pass threshold [mS/cm]
    alk_conc: float = 0.02  # NaOH concentration [fraction]
    acid_conc: float = 0.01 # AcAc concentration [fraction]


RECIPES = {
    1: Recipe('STD-DESCALE',
              alk_temp_sp=30.0, acid_temp_sp=40.0,
              dur_init_rinse=180,   dur_alk_wash=2520,
              dur_inter_rinse=180,  dur_acid_wash=1620,
              dur_final_rinse=300,  cond_gate=0.5),
    2: Recipe('HEAVY-SCALE',
              alk_temp_sp=35.0, acid_temp_sp=45.0,
              dur_init_rinse=180,   dur_alk_wash=3000,
              dur_inter_rinse=240,  dur_acid_wash=1800,
              dur_final_rinse=300,  cond_gate=0.5),
    3: Recipe('LIGHT-MAINT',
              alk_temp_sp=25.0, acid_temp_sp=35.0,
              dur_init_rinse=120,   dur_alk_wash=1800,
              dur_inter_rinse=120,  dur_acid_wash=1200,
              dur_final_rinse=300,  cond_gate=0.5),
    4: Recipe('PHARMA-GRADE',
              alk_temp_sp=30.0, acid_temp_sp=40.0,
              dur_init_rinse=240,   dur_alk_wash=2700,
              dur_inter_rinse=240,  dur_acid_wash=1800,
              dur_final_rinse=600,  cond_gate=0.3),
}


@dataclass
class AlarmRecord:
    timestamp: float
    tag: str
    description: str
    priority: int
    acknowledged: bool = False


class PIDController:
    """
    ISA parallel PID with:
      - Back-calculation anti-windup (Kb = 1/Ti)
      - First-order derivative filter on PV (prevents setpoint kick)
      - Bumpless auto/manual transfer
      - Deadband on error
      - Output saturation
    """

    def __init__(self, Kp: float = 3.5, Ti: float = 12.5,
                 Td: float = 6.25, Tf: float = 0.5,
                 dt: float = 0.01, out_min: float = 0.0,
                 out_max: float = 100.0, deadband: float = 0.2):
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.Tf = Tf
        self.dt = dt
        self.out_min = out_min
        self.out_max = out_max
        self.deadband = deadband

        # Internal state
        self._integral   = 0.0
        self._d_filtered = 0.0
        self._prev_pv    = 0.0
        self._back_calc  = 0.0

        # Outputs
        self.output   = 0.0
        self.error    = 0.0
        self.p_term   = 0.0
        self.i_term   = 0.0
        self.d_term   = 0.0
        self.sat_hi   = False
        self.sat_lo   = False

    def update(self, sp: float, pv: float, auto: bool,
               manual_out: float = 0.0) -> float:
        if not auto:
            # Bumpless transfer: pre-load integrator
            self.output     = max(self.out_min, min(self.out_max, manual_out))
            self._integral  = self.output
            self._prev_pv   = pv
            self._d_filtered = 0.0
            return self.output

        # Derived gains
        Ki = (1.0 / self.Ti) * self.dt
        Kb = self.dt / self.Ti

        # Error with deadband
        self.error = sp - pv
        eff_error  = 0.0 if abs(self.error) <= self.deadband else self.error

        # Proportional
        self.p_term = self.Kp * eff_error

        # Integral with back-calculation AWU
        self._integral += self.Kp * Ki * eff_error + self._back_calc
        self.i_term     = self._integral

        # Derivative on PV (filtered)
        self._d_filtered = (
            self.Td * (self._prev_pv - pv) / self.dt
            + self._d_filtered * self.Tf
        ) / (self.Tf + self.dt)
        self.d_term = self.Kp * self._d_filtered

        # Sum
        raw_out = self.p_term + self.i_term + self.d_term

        # Saturation
        self.sat_hi = raw_out > self.out_max
        self.sat_lo = raw_out < self.out_min

        if raw_out > self.out_max:
            self.output = self.out_max
        elif raw_out < self.out_min:
            self.output = self.out_min
        else:
            self.output = raw_out

        # Back-calculation
        self._back_calc = Kb * (self.output - raw_out)
        self._prev_pv   = pv

        return self.output

    def reset(self):
        self._integral   = 0.0
        self._d_filtered = 0.0
        self._prev_pv    = 0.0
        self._back_calc  = 0.0
        self.output      = 0.0



class ThermalModel:
    """
    First-Order-Plus-Dead-Time (FOPDT) thermal process model.

    Transfer function:  G(s) = K * exp(-θs) / (τs + 1)

    Parameters calibrated from Skikda boiler thermal response:
      K  = 0.80 °C / % valve open   (process gain)
      τ  = 120 s                     (time constant)
      θ  = 5  s                     (dead time — transport lag)

    Ambient heat loss modelled as first-order decay toward T_ambient.
    """

    def __init__(self, K: float = 0.80, tau: float = 120.0,
                 theta: float = 5.0, T_ambient: float = 22.0,
                 dt: float = 0.1, noise_std: float = 0.05):
        self.K         = K
        self.tau       = tau
        self.theta     = theta
        self.T_ambient = T_ambient
        self.dt        = dt
        self.noise_std = noise_std

        # Dead-time buffer (number of steps for dead time)
        self._n_dead = max(1, int(theta / dt))
        self._valve_buf = [0.0] * (self._n_dead + 1)

        # State
        self.T = T_ambient    # Current temperature [°C]
        self.rng = np.random.default_rng(42)

    def step(self, valve_pct: float, steam_enable: bool) -> float:
        """Advance one time step. Returns new temperature."""
        # Update dead-time FIFO buffer
        self._valve_buf.append(valve_pct if steam_enable else 0.0)
        delayed_valve = self._valve_buf.pop(0)

        # Continuous-time ODE discretised with Euler:
        # dT/dt = (K * u - (T - T_ambient)) / tau
        dT_dt = (self.K * delayed_valve - (self.T - self.T_ambient)) / self.tau
        self.T += dT_dt * self.dt

        # Add measurement noise
        noise = self.rng.normal(0, self.noise_std)
        return np.clip(self.T + noise, -5.0, 120.0)


class ConductivityModel:
    """
    Conductivity of the circuit fluid over time.

    - During chemical phases: conductivity rises as scale dissolves
    - During rinse phases: exponential decay toward rinse-water baseline
    - Initial value depends on boiler water hardness
    """

    def __init__(self, initial_cond: float = 8.0,
                 rinse_baseline: float = 0.15,
                 decay_tau: float = 45.0,
                 dt: float = 0.1, noise_std: float = 0.01):
        self.cond          = initial_cond
        self.rinse_baseline = rinse_baseline
        self.decay_tau     = decay_tau
        self.dt            = dt
        self.noise_std     = noise_std
        self.rng           = np.random.default_rng(99)

    def step(self, state: CIPState, flow_active: bool,
             alk_valve: bool, acid_valve: bool,
             rinse_valve: bool) -> float:

        if not flow_active:
            # No flow: slow evaporation concentration effect
            self.cond += 0.0005 * self.dt
        elif rinse_valve and not alk_valve and not acid_valve:
            # Rinse phase: exponential decay toward baseline
            dcond_dt = -(self.cond - self.rinse_baseline) / self.decay_tau
            self.cond += dcond_dt * self.dt
        elif alk_valve:
            # Alkaline wash: conductivity increases as NaOH dissolves scale
            if self.cond < 15.0:
                self.cond += 0.008 * self.dt
        elif acid_valve:
            # Acid wash: different conductivity signature of dissolving CaCO3
            # CO2 evolution creates transient increase then decrease
            if self.cond < 12.0:
                self.cond += 0.012 * self.dt

        noise = self.rng.normal(0, self.noise_std)
        return max(0.05, self.cond + noise)


class FlowModel:
    """
    Simple hydraulic model for recirculation flow.
    Pump power → flow through circuit resistance.
    Valve states affect circuit resistance.
    """

    def __init__(self, nominal_flow: float = 25.0,
                 dt: float = 0.1, noise_std: float = 0.3):
        self.nominal_flow = nominal_flow
        self.dt           = dt
        self.noise_std    = noise_std
        self.flow         = 0.0
        self.rng          = np.random.default_rng(7)

    def step(self, pump_on: bool, drain_open: bool) -> float:
        if not pump_on:
            # Ramp down
            self.flow = max(0.0, self.flow - 5.0 * self.dt)
        else:
            # Ramp up toward nominal, reduced if drain open (bypass flow)
            target = self.nominal_flow * (0.7 if drain_open else 1.0)
            self.flow += (target - self.flow) * 0.15 * self.dt
            self.flow  = max(0.0, self.flow)

        noise = self.rng.normal(0, self.noise_std)
        return max(0.0, self.flow + noise)



class CIPStateMachine:
    """
    Complete ISA-88 CIP state machine, exact behavioural replica
    of the CODESYS FB_CIP function block.
    """

    def __init__(self, recipe: Recipe, dt: float = 0.1):
        self.recipe  = recipe
        self.dt      = dt

        # State
        self.state   = CIPState.IDLE
        self.phase_timer   = 0.0
        self.phase_elapsed = 0.0
        self.progress      = 0.0

        # Outputs
        self.q_pump  = False
        self.q_naoh  = False
        self.q_acac  = False
        self.q_rinse = False
        self.q_drain = False
        self.q_steam = False
        self.temp_sp_active = 0.0

        # Accumulators
        self.naoh_vol = 0.0   # [L]
        self.acac_vol = 0.0   # [L]

        # Alarms
        self.alarm_rinse_fail = False

        # Hold memory
        self._hold_resume_state = CIPState.IDLE

        # State entrance flag
        self._state_entered = False
        self._prev_state    = CIPState.IDLE

        # Transition log
        self.transition_log: List[Tuple[float, CIPState, CIPState]] = []
        self.time_total = 0.0

        # Permissive (set externally)
        self.permissive_ok = True

        # Commands (set externally each step)
        self.cmd_start   = False
        self.cmd_stop    = False
        self.cmd_hold    = False
        self.cmd_restart = False
        self.cmd_abort   = False
        self.cmd_reset   = False

        # Safety flags (set externally)
        self.estop_active = False

    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(self.state, f'UNKNOWN({self.state})')

    def _transition(self, new_state: CIPState):
        self.transition_log.append((self.time_total, self.state, new_state))
        self._prev_state  = self.state
        self.state        = new_state
        self.phase_timer  = 0.0
        self.phase_elapsed = 0.0

    def _all_off(self):
        self.q_pump = self.q_naoh = self.q_acac = False
        self.q_rinse = self.q_drain = self.q_steam = False
        self.temp_sp_active = 0.0

    def step(self, flow_pv: float, cond_pv: float,
             temp_pv: float) -> None:
        """Advance state machine by one time step."""
        self.time_total += self.dt

        # ── ABORT — highest priority ─────────────────────────
        if self.cmd_abort and self.state not in (CIPState.ABORTING,
                                                  CIPState.ABORTED):
            self._all_off()
            self.q_drain = True
            self._transition(CIPState.ABORTING)
            return

        # ── E-STOP ───────────────────────────────────────────
        if self.estop_active:
            self._all_off()
            self.q_drain = True
            self.state = CIPState.ESTOP
            return

        # ── TIMER ADVANCE ────────────────────────────────────
        self.phase_timer += self.dt

        # ──────────────────────────────────────────────────────
        # MAIN STATE DISPATCH
        # ──────────────────────────────────────────────────────
        s = self.state

        if s == CIPState.IDLE:
            self._all_off()
            self.progress = 0.0
            if self.cmd_start and self.permissive_ok:
                self._transition(CIPState.PRECHECK)

        elif s == CIPState.PRECHECK:
            self.q_pump  = True
            self.q_rinse = True
            # 30 second pre-check: validate all permissives
            if self.phase_timer >= 30.0:
                if self.permissive_ok and flow_pv > 5.0:
                    self._transition(CIPState.INIT_RINSE)
                else:
                    self._transition(CIPState.ABORTING)

        elif s == CIPState.INIT_RINSE:
            self.q_pump  = True
            self.q_rinse = True
            self.q_drain = True
            self.q_steam = False
            self.q_naoh  = False
            self.q_acac  = False
            self.temp_sp_active = 0.0
            target = self.recipe.dur_init_rinse
            self.progress = min(100.0, self.phase_timer / target * 100.0)
            if self.phase_timer >= target:
                self.q_drain = False
                self._transition(CIPState.ALK_WASH)

        elif s == CIPState.ALK_WASH:
            self.q_pump  = True
            self.q_naoh  = True
            self.q_steam = True
            self.q_rinse = False
            self.q_acac  = False
            self.q_drain = False
            self.temp_sp_active = self.recipe.alk_temp_sp
            target = self.recipe.dur_alk_wash
            self.progress = min(100.0, self.phase_timer / target * 100.0)
            # Accumulate NaOH volume (flow [L/min] * conc * dt/60)
            self.naoh_vol += flow_pv * self.recipe.alk_conc * self.dt / 60.0
            if self.cmd_hold or not self.permissive_ok:
                self._hold_resume_state = CIPState.ALK_WASH
                self._transition(CIPState.HOLDING)
            elif self.cmd_stop:
                self._transition(CIPState.STOPPING)
            elif self.phase_timer >= target:
                self.q_steam = False
                self._transition(CIPState.INTER_RINSE)

        elif s == CIPState.INTER_RINSE:
            self.q_pump  = True
            self.q_rinse = True
            self.q_drain = True
            self.q_naoh  = False
            self.q_acac  = False
            self.q_steam = False
            self.temp_sp_active = 0.0
            target = self.recipe.dur_inter_rinse
            self.progress = min(100.0, self.phase_timer / target * 100.0)
            if self.phase_timer >= target:
                self.q_drain = False
                self._transition(CIPState.ACID_WASH)

        elif s == CIPState.ACID_WASH:
            self.q_pump  = True
            self.q_acac  = True
            self.q_steam = True
            self.q_rinse = False
            self.q_naoh  = False
            self.q_drain = False
            self.temp_sp_active = self.recipe.acid_temp_sp
            target = self.recipe.dur_acid_wash
            self.progress = min(100.0, self.phase_timer / target * 100.0)
            self.acac_vol += flow_pv * self.recipe.acid_conc * self.dt / 60.0
            if self.cmd_hold or not self.permissive_ok:
                self._hold_resume_state = CIPState.ACID_WASH
                self._transition(CIPState.HOLDING)
            elif self.cmd_stop:
                self._transition(CIPState.STOPPING)
            elif self.phase_timer >= target:
                self.q_steam = False
                self._transition(CIPState.FINAL_RINSE)

        elif s == CIPState.FINAL_RINSE:
            self.q_pump  = True
            self.q_rinse = True
            self.q_drain = True
            self.q_acac  = False
            self.q_steam = False
            self.temp_sp_active = 0.0
            target = self.recipe.dur_final_rinse
            self.progress = min(100.0, self.phase_timer / target * 100.0)
            if self.phase_timer >= target:
                if cond_pv <= self.recipe.cond_gate:
                    self._transition(CIPState.COMPLETE)
                else:
                    self.alarm_rinse_fail = True
                    self._transition(CIPState.ABORTING)

        elif s == CIPState.COMPLETE:
            self._all_off()
            self.progress = 100.0
            if self.cmd_reset:
                self.naoh_vol = 0.0
                self.acac_vol = 0.0
                self._transition(CIPState.IDLE)

        elif s == CIPState.HOLDING:
            self.q_pump  = True
            self.q_rinse = True
            self.q_naoh  = False
            self.q_acac  = False
            self.q_steam = False
            # After 5s stabilise → HELD
            if self.phase_timer >= 5.0:
                self._transition(CIPState.HELD)

        elif s == CIPState.HELD:
            self.q_pump  = True
            self.q_rinse = True
            if self.cmd_restart and self.permissive_ok:
                self._transition(CIPState.RESTARTING)

        elif s == CIPState.RESTARTING:
            self.q_pump  = True
            self.q_rinse = True
            if self.phase_timer >= 5.0 and self.permissive_ok:
                self.state = self._hold_resume_state
                self.phase_timer = 0.0

        elif s == CIPState.STOPPING:
            self.q_naoh  = False
            self.q_acac  = False
            self.q_steam = False
            self.q_pump  = True
            self.q_rinse = True
            if self.phase_timer >= 30.0:
                self.q_pump  = False
                self.q_rinse = False
                self._transition(CIPState.STOPPED)

        elif s == CIPState.STOPPED:
            self._all_off()
            if self.cmd_reset:
                self._transition(CIPState.IDLE)

        elif s == CIPState.ABORTING:
            self._all_off()
            self.q_drain = True
            # 30s drain
            if self.phase_timer >= 30.0:
                self.q_drain = False
                self._transition(CIPState.ABORTED)

        elif s == CIPState.ABORTED:
            self._all_off()
            if self.cmd_reset:
                self.alarm_rinse_fail = False
                self._transition(CIPState.IDLE)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — ALARM MANAGER
# ═══════════════════════════════════════════════════════════════════════

class AlarmManager:
    """ISA-18.2 alarm management with latching, ACK, and priority."""

    PRIORITY_LABELS = {1: 'CRITICAL', 2: 'HIGH', 3: 'MEDIUM', 4: 'LOW'}
    PRIORITY_COLORS = {1: '#ff3b3b', 2: '#ffc845', 3: '#ffee58', 4: '#90caf9'}

    def __init__(self):
        self.alarms: List[AlarmRecord] = []
        self._prev_states: dict = {}

    def evaluate(self, tag: str, condition: bool, description: str,
                 priority: int, timestamp: float):
        prev = self._prev_states.get(tag, False)
        # Rising edge
        if condition and not prev:
            self.alarms.append(AlarmRecord(
                timestamp=timestamp,
                tag=tag,
                description=description,
                priority=priority,
            ))
        self._prev_states[tag] = condition

    def acknowledge(self, tag: str):
        for a in self.alarms:
            if a.tag == tag:
                a.acknowledged = True

    def acknowledge_all(self):
        for a in self.alarms:
            a.acknowledged = True

    @property
    def unacknowledged_count(self) -> int:
        return sum(1 for a in self.alarms if not a.acknowledged)

    @property
    def highest_priority_active(self) -> Optional[int]:
        active = [a.priority for a in self.alarms if not a.acknowledged]
        return min(active) if active else None


class DisturbanceInjector:
    """
    Injects realistic process disturbances for testing robustness.
    Includes: steam valve sticking, temperature sensor drift,
    flow transient, conductivity spike.
    """

    def __init__(self, enabled: bool = False, seed: int = 42):
        self.enabled = enabled
        self.rng     = np.random.default_rng(seed)
        self._events_fired: set = set()

    def apply_valve_disturbance(self, valve_pct: float,
                                 sim_time: float) -> float:
        if not self.enabled:
            return valve_pct
        # At t=1800s: steam valve sticks at 75% for 60 seconds
        if 1800 <= sim_time <= 1860 and 'valve_stick' not in self._events_fired:
            self._events_fired.add('valve_stick')
            return 75.0
        return valve_pct

    def apply_temp_disturbance(self, temp: float,
                                sim_time: float) -> float:
        if not self.enabled:
            return temp
        # Random walk noise on top of model noise
        return temp + self.rng.normal(0, 0.08)

    def apply_flow_disturbance(self, flow: float,
                                sim_time: float) -> float:
        if not self.enabled:
            return flow
        # At t=3000s: brief flow transient (30s)
        if 3000 <= sim_time <= 3030:
            return max(0.0, flow - 8.0)
        return flow


class CIPSimulation:
    """
    Master simulation coordinator. Wires together all sub-models,
    runs the control loop, and records data for plotting.
    """

    DT_MODEL    = 0.1    # Model integration step [s]
    DT_PLOT     = 1.0    # Plot refresh step [s] (every N model steps)
    DT_CTRL     = 0.01   # PID execution period [s] — matches 10ms task

    def __init__(self, recipe_id: int = 1, inject_disturbance: bool = False):
        self.recipe   = RECIPES[recipe_id]
        self.dt       = self.DT_MODEL

        # Sub-models
        self.thermal    = ThermalModel(dt=self.dt)
        self.cond_model = ConductivityModel(dt=self.dt)
        self.flow_model = FlowModel(dt=self.dt)
        self.pid        = PIDController(Kp=3.5, Ti=12.5, Td=6.25, Tf=0.5,
                                         dt=self.DT_CTRL)
        self.state_machine = CIPStateMachine(recipe=self.recipe, dt=self.dt)
        self.alarms     = AlarmManager()
        self.disturb    = DisturbanceInjector(enabled=inject_disturbance)

        # Process values
        self.temp_pv   = 22.0
        self.cond_pv   = 8.0
        self.flow_pv   = 0.0
        self.pres_pv   = 1.0   # Approximate, simplified
        self.steam_pct = 0.0

        # Recorded data (for plotting)
        self.hist_t:         List[float] = []
        self.hist_temp_pv:   List[float] = []
        self.hist_temp_sp:   List[float] = []
        self.hist_steam:     List[float] = []
        self.hist_cond:      List[float] = []
        self.hist_flow:      List[float] = []
        self.hist_state:     List[int]   = []
        self.hist_naoh_vol:  List[float] = []
        self.hist_acac_vol:  List[float] = []
        self.hist_pid_p:     List[float] = []
        self.hist_pid_i:     List[float] = []
        self.hist_pid_d:     List[float] = []

        # Control flags
        self.running    = False
        self.paused     = False
        self.sim_speed  = 60.0  # 1 real second = 60 simulation seconds

        # Batch record
        self.batch_id    = time.strftime('%Y%m%d_%H%M%S')
        self.batch_start = 0.0

        # Safety
        self.TEMP_TRIP  = 60.0
        self.PRES_TRIP  = 6.0

    def _evaluate_safety(self) -> bool:
        """Returns True if EStop should be active."""
        t = self.state_machine.time_total
        self.alarms.evaluate('HIGH_TEMP', self.temp_pv >= self.TEMP_TRIP,
                              'High temperature trip', 1, t)
        self.alarms.evaluate('FLOW_FAULT',
                              self.state_machine.q_pump and self.flow_pv < 2.0,
                              'Pump / low flow fault', 2, t)
        self.alarms.evaluate('RINSE_FAIL', self.state_machine.alarm_rinse_fail,
                              'Final rinse conductivity fail', 2, t)
        return self.temp_pv >= self.TEMP_TRIP

    def step(self):
        """Single simulation time step."""
        sm = self.state_machine

        # ── Safety ──────────────────────────────────────────
        sm.estop_active = self._evaluate_safety()

        # ── State machine advancement ────────────────────────
        sm.step(self.flow_pv, self.cond_pv, self.temp_pv)

        # ── PID controller ───────────────────────────────────
        pid_auto = sm.q_steam and not sm.estop_active
        self.steam_pct = self.pid.update(
            sp=sm.temp_sp_active,
            pv=self.temp_pv,
            auto=pid_auto
        )

        # ── Disturbance injection ────────────────────────────
        valve_in = self.disturb.apply_valve_disturbance(
            self.steam_pct, sm.time_total)

        # ── Process models ───────────────────────────────────
        self.temp_pv = self.thermal.step(valve_in, sm.q_steam)
        self.temp_pv = self.disturb.apply_temp_disturbance(
            self.temp_pv, sm.time_total)

        self.flow_pv = self.flow_model.step(sm.q_pump, sm.q_drain)
        self.flow_pv = self.disturb.apply_flow_disturbance(
            self.flow_pv, sm.time_total)

        self.cond_pv = self.cond_model.step(
            sm.state, sm.q_pump, sm.q_naoh, sm.q_acac, sm.q_rinse)

        # ── Permissive update ─────────────────────────────────
        sm.permissive_ok = (
            self.flow_pv > 5.0
            and not sm.estop_active
            and self.temp_pv < (self.TEMP_TRIP - 10.0)
        )

        # ── Record ───────────────────────────────────────────
        t = sm.time_total
        self.hist_t.append(t)
        self.hist_temp_pv.append(self.temp_pv)
        self.hist_temp_sp.append(sm.temp_sp_active)
        self.hist_steam.append(self.steam_pct)
        self.hist_cond.append(self.cond_pv)
        self.hist_flow.append(self.flow_pv)
        self.hist_state.append(int(sm.state))
        self.hist_naoh_vol.append(sm.naoh_vol)
        self.hist_acac_vol.append(sm.acac_vol)
        self.hist_pid_p.append(self.pid.p_term)
        self.hist_pid_i.append(self.pid.i_term)
        self.hist_pid_d.append(self.pid.d_term)

    def auto_start(self):
        """Send START command after 5 seconds."""
        pass  # Handled in run loop

    def get_batch_report(self) -> dict:
        """Generate end-of-batch summary statistics."""
        if not self.hist_t:
            return {}

        sm = self.state_machine
        alk_mask  = [s == int(CIPState.ALK_WASH) for s in self.hist_state]
        acid_mask = [s == int(CIPState.ACID_WASH) for s in self.hist_state]

        alk_temps  = [t for t, m in zip(self.hist_temp_pv, alk_mask) if m]
        acid_temps = [t for t, m in zip(self.hist_temp_pv, acid_mask) if m]

        return {
            'batch_id':        self.batch_id,
            'recipe':          self.recipe.name,
            'total_duration_s':self.hist_t[-1],
            'alk_temp_avg':    np.mean(alk_temps) if alk_temps else 0,
            'alk_temp_max':    np.max(alk_temps)  if alk_temps else 0,
            'acid_temp_avg':   np.mean(acid_temps) if acid_temps else 0,
            'acid_temp_max':   np.max(acid_temps)  if acid_temps else 0,
            'final_cond_pv':   self.cond_pv,
            'cond_pass':       self.cond_pv <= self.recipe.cond_gate,
            'naoh_vol_L':      sm.naoh_vol,
            'acac_vol_L':      sm.acac_vol,
            'alarm_count':     len(self.alarms.alarms),
            'state_final':     sm.state_name,
        }



class CIPDashboard:
    """
    Live Matplotlib dashboard — 6-panel layout:
      [0] Temperature PV + SP + Steam valve
      [1] Conductivity + gate threshold
      [2] Flow rate + minimum permissive line
      [3] PID P/I/D component trends
      [4] State machine timeline (phase shading)
      [5] Chemical accumulation + status panel
    """

    PHASE_COLORS = {
        CIPState.INIT_RINSE:  ('#1a3020', '#69ff9a'),
        CIPState.ALK_WASH:    ('#2a1f00', '#ffc845'),
        CIPState.INTER_RINSE: ('#1a3020', '#69ff9a'),
        CIPState.ACID_WASH:   ('#2a0a0a', '#ff6b6b'),
        CIPState.FINAL_RINSE: ('#0a1a2a', '#4de8ff'),
    }

    def __init__(self, sim: CIPSimulation):
        self.sim = sim
        self._init_figure()

    def _init_figure(self):
        self.fig = plt.figure(figsize=(18, 11))
        self.fig.patch.set_facecolor('#0a0c10')
        plt.suptitle(
            f'CIP BOILER DESCALING — DIGITAL TWIN  |  Recipe: '
            f'{self.sim.recipe.name}  |  Batch: {self.sim.batch_id}',
            fontsize=11, fontweight='bold',
            color='#ffc845', y=0.98, fontfamily='monospace'
        )

        gs = gridspec.GridSpec(3, 3, figure=self.fig,
                               hspace=0.42, wspace=0.32,
                               left=0.06, right=0.97,
                               top=0.94, bottom=0.05)

        # Panel definitions
        self.ax_temp   = self.fig.add_subplot(gs[0, 0:2])
        self.ax_steam  = self.ax_temp.twinx()
        self.ax_cond   = self.fig.add_subplot(gs[1, 0:2])
        self.ax_flow   = self.fig.add_subplot(gs[2, 0:2])
        self.ax_pid    = self.fig.add_subplot(gs[0, 2])
        self.ax_state  = self.fig.add_subplot(gs[1, 2])
        self.ax_chem   = self.fig.add_subplot(gs[2, 2])

        # Configure axes
        panels = [self.ax_temp, self.ax_cond, self.ax_flow,
                  self.ax_pid, self.ax_state, self.ax_chem]
        titles = ['TEMPERATURE [°C]', 'CONDUCTIVITY [mS/cm]',
                  'FLOW [L/min]', 'PID COMPONENTS [%]',
                  'CIP STATE', 'CHEMICAL CONSUMPTION [L]']
        for ax, title in zip(panels, titles):
            ax.set_title(title, fontsize=8, color='#5a6a80',
                         loc='left', pad=4, fontfamily='monospace')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

        self.ax_steam.set_ylabel('Steam %', fontsize=7, color='#ff9a6b')
        self.ax_steam.tick_params(labelsize=7, colors='#ff9a6b')

        # Initialize line objects
        self.line_temp_pv, = self.ax_temp.plot([], [], '#4de8ff',
                                                lw=1.8, label='Temp PV')
        self.line_temp_sp, = self.ax_temp.plot([], [], '#ff6b6b',
                                                lw=1.2, ls='--', label='SP')
        self.line_steam,   = self.ax_steam.plot([], [], '#ff9a6b',
                                                  lw=1.0, alpha=0.7,
                                                  label='Steam %')
        self.line_cond,    = self.ax_cond.plot([], [], '#ffc845',
                                                lw=1.8, label='Cond PV')
        self.line_flow,    = self.ax_flow.plot([], [], '#69ff9a',
                                                lw=1.8, label='Flow')
        self.line_pid_p,   = self.ax_pid.plot([], [], '#4de8ff',
                                               lw=1.2, label='P')
        self.line_pid_i,   = self.ax_pid.plot([], [], '#ffc845',
                                               lw=1.2, label='I')
        self.line_pid_d,   = self.ax_pid.plot([], [], '#ff6b6b',
                                               lw=1.2, label='D')
        self.line_naoh,    = self.ax_chem.plot([], [], '#ffc845',
                                                lw=1.8, label='NaOH (L)')
        self.line_acac,    = self.ax_chem.plot([], [], '#ff6b6b',
                                                lw=1.8, label='AcAc (L)')

        # Reference lines
        self.ax_cond.axhline(self.sim.recipe.cond_gate,
                              color='#ff3b3b', lw=1.0, ls='--',
                              label=f'Gate {self.sim.recipe.cond_gate}')
        self.ax_flow.axhline(5.0, color='#ff6b6b', lw=1.0, ls='--',
                              label='Min 5 L/min')
        self.ax_temp.set_ylim(15, 65)
        self.ax_steam.set_ylim(-5, 115)
        self.ax_cond.set_ylim(0, 16)
        self.ax_flow.set_ylim(-2, 35)
        self.ax_pid.set_ylim(-30, 110)

        # Legends
        self.ax_temp.legend(loc='upper left', fontsize=7,
                             facecolor='#0f1318', edgecolor='#2a3547')
        self.ax_cond.legend(loc='upper right', fontsize=7,
                             facecolor='#0f1318', edgecolor='#2a3547')
        self.ax_pid.legend(loc='upper right', fontsize=7,
                            facecolor='#0f1318', edgecolor='#2a3547')
        self.ax_chem.legend(loc='upper left', fontsize=7,
                             facecolor='#0f1318', edgecolor='#2a3547')

        # Status text box
        self.status_text = self.ax_state.text(
            0.5, 0.5, 'IDLE', ha='center', va='center',
            fontsize=18, fontweight='bold', color='#5a6a80',
            transform=self.ax_state.transAxes, fontfamily='monospace'
        )
        self.state_info_text = self.ax_state.text(
            0.5, 0.25, '', ha='center', va='center',
            fontsize=8, color='#5a6a80',
            transform=self.ax_state.transAxes, fontfamily='monospace'
        )
        self.ax_state.set_xlim(0, 1)
        self.ax_state.set_ylim(0, 1)
        self.ax_state.set_xticks([])
        self.ax_state.set_yticks([])

        # Alarm text
        self.alarm_text = self.fig.text(
            0.5, 0.007, '', ha='center', fontsize=8,
            color='#ff3b3b', fontfamily='monospace', fontweight='bold'
        )

    def _shade_phases(self, ax, t_data, state_data):
        """Shade background by CIP phase."""
        ax.cla()

    def update(self, frame: int) -> list:
        """Matplotlib animation update callback."""
        sim = self.sim

        # Run N simulation steps per frame
        steps_per_frame = max(1, int(self.sim.sim_speed / 10))
        for _ in range(steps_per_frame):
            sim.step()
            # Auto-start after 5 sim seconds
            if sim.state_machine.time_total >= 5.0 and \
               sim.state_machine.state == CIPState.IDLE:
                sim.state_machine.cmd_start = True
            else:
                sim.state_machine.cmd_start = False

            # Auto-reset after complete
            if sim.state_machine.state in (CIPState.COMPLETE,
                                            CIPState.ABORTED,
                                            CIPState.STOPPED):
                # Stop animation — batch done
                pass

        t    = sim.hist_t
        tmin = t[-1] - 3600 if len(t) > 1 else 0  # rolling 60min window

        def rolling(data):
            mask = [ti >= tmin for ti in t]
            tt   = [ti for ti, m in zip(t, mask) if m]
            dd   = [d  for d,  m in zip(data, mask) if m]
            return tt, dd

        rt, rd = rolling(sim.hist_temp_pv)
        self.line_temp_pv.set_data(rt, rd)
        rt, rd = rolling(sim.hist_temp_sp)
        self.line_temp_sp.set_data(rt, rd)
        rt, rd = rolling(sim.hist_steam)
        self.line_steam.set_data(rt, rd)
        rt, rd = rolling(sim.hist_cond)
        self.line_cond.set_data(rt, rd)
        rt, rd = rolling(sim.hist_flow)
        self.line_flow.set_data(rt, rd)
        rt, rd = rolling(sim.hist_pid_p)
        self.line_pid_p.set_data(rt, rd)
        rt, rd = rolling(sim.hist_pid_i)
        self.line_pid_i.set_data(rt, rd)
        rt, rd = rolling(sim.hist_pid_d)
        self.line_pid_d.set_data(rt, rd)

        # Chemical accumulation — full history
        self.line_naoh.set_data(t, sim.hist_naoh_vol)
        self.line_acac.set_data(t, sim.hist_acac_vol)

        # X-axis autoscale
        if len(rt) > 2:
            x_lo, x_hi = rt[0], rt[-1]
            for ax in [self.ax_temp, self.ax_cond, self.ax_flow, self.ax_pid]:
                ax.set_xlim(x_lo, x_hi)
        if t:
            self.ax_chem.set_xlim(0, max(t[-1], 100))
            mv = max(max(sim.hist_naoh_vol, default=0),
                     max(sim.hist_acac_vol, default=0), 1)
            self.ax_chem.set_ylim(0, mv * 1.2)

        # State display
        sm = sim.state_machine
        state_name  = sm.state_name
        state_color = STATE_COLORS.get(sm.state, '#5a6a80')
        self.status_text.set_text(state_name)
        self.status_text.set_color(state_color)

        elapsed_min = sm.time_total / 60.0
        info = (f'T_total: {elapsed_min:.1f} min\n'
                f'Progress: {sm.progress:.1f}%\n'
                f'Temp: {sim.temp_pv:.1f}°C\n'
                f'Cond: {sim.cond_pv:.3f} mS/cm\n'
                f'Flow: {sim.flow_pv:.1f} L/min\n'
                f'Steam: {sim.steam_pct:.1f}%')
        self.state_info_text.set_text(info)
        self.state_info_text.set_color('#8a9ab0')

        # Background colour follows state
        bg = '#0f1318'
        if sm.state == CIPState.ALK_WASH:
            bg = '#1a1500'
        elif sm.state == CIPState.ACID_WASH:
            bg = '#1a0a0a'
        elif sm.state == CIPState.FINAL_RINSE:
            bg = '#0a1218'
        elif sm.estop_active:
            bg = '#1a0000'
        self.ax_state.set_facecolor(bg)

        # Alarm display
        n_alarms = sim.alarms.unacknowledged_count
        if n_alarms > 0:
            alm_txt = f'⚠ {n_alarms} UNACKNOWLEDGED ALARM(S)'
            self.alarm_text.set_text(alm_txt)
        else:
            self.alarm_text.set_text('')

        return (self.line_temp_pv, self.line_temp_sp, self.line_steam,
                self.line_cond, self.line_flow, self.line_pid_p,
                self.line_pid_i, self.line_pid_d, self.line_naoh,
                self.line_acac, self.status_text, self.state_info_text)



def run_static_simulation(recipe_id: int, inject_disturbance: bool) -> None:
    """
    Run complete batch headlessly, then plot full results.
    Useful for CI/batch analysis without real-time display.
    """
    print(f"\n{'='*70}")
    print(f"  CIP DIGITAL TWIN — STATIC SIMULATION")
    print(f"  Recipe: {RECIPES[recipe_id].name}  |  "
          f"Disturbances: {'ON' if inject_disturbance else 'OFF'}")
    print(f"{'='*70}")

    sim = CIPSimulation(recipe_id=recipe_id,
                        inject_disturbance=inject_disturbance)
    sim.sim_speed = 1.0  # Real-time speed irrelevant for static run

    sm = sim.state_machine
    total_recipe_duration = (
        sim.recipe.dur_init_rinse  +
        sim.recipe.dur_alk_wash    +
        sim.recipe.dur_inter_rinse +
        sim.recipe.dur_acid_wash   +
        sim.recipe.dur_final_rinse +
        30.0  # pre-check
    ) + 60.0  # buffer

    step_count = int(total_recipe_duration / sim.dt)
    print(f"  Simulating {total_recipe_duration/60:.1f} min "
          f"({step_count:,} steps at dt={sim.dt}s)...")

    sm.cmd_start = False
    auto_started = False
    last_state   = sm.state
    t_start_wall = time.time()

    for i in range(step_count):
        sim.step()

        # Auto-start at t=5s
        if sm.time_total >= 5.0 and not auto_started:
            sm.cmd_start = True
            auto_started = True
        else:
            sm.cmd_start = False

        # State change logging
        if sm.state != last_state:
            mins = sm.time_total / 60.0
            print(f"  [{mins:6.1f} min] ─ {STATE_NAMES[last_state]:<15} → "
                  f"{STATE_NAMES[sm.state]}")
            last_state = sm.state

        # Stop after complete/aborted
        if sm.state in (CIPState.COMPLETE, CIPState.ABORTED,
                         CIPState.STOPPED) and sm.phase_timer > 5.0:
            break

    elapsed_wall = time.time() - t_start_wall
    print(f"\n  Simulation completed in {elapsed_wall:.2f}s wall time")

    # ── Print batch report ──────────────────────────────────
    report = sim.get_batch_report()
    print(f"\n{'─'*50}")
    print(f"  BATCH REPORT — {report['batch_id']}")
    print(f"{'─'*50}")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k:<25} {v:.3f}")
        else:
            print(f"  {k:<25} {v}")

    # ── Alarm summary ───────────────────────────────────────
    if sim.alarms.alarms:
        print(f"\n  ALARM LOG ({len(sim.alarms.alarms)} events):")
        for a in sim.alarms.alarms:
            mins = a.timestamp / 60.0
            print(f"  [{mins:6.1f} min] P{a.priority} {a.tag:<20} "
                  f"{a.description}")
    else:
        print("\n  ALARM LOG: No alarms during batch. ✓")

    # ── Static plot ─────────────────────────────────────────
    _plot_static_results(sim)


def _plot_static_results(sim: CIPSimulation):
    t = np.array(sim.hist_t) / 60.0  # minutes

    fig, axes = plt.subplots(4, 1, figsize=(16, 12),
                              sharex=True)
    fig.suptitle(
        f'CIP BATCH REPORT — {sim.recipe.name} — {sim.batch_id}',
        fontsize=12, color='#ffc845', fontweight='bold', y=0.98
    )

    # ── Shade phases ──────────────────────────────────────
    state_arr = np.array(sim.hist_state)
    for phase, (bg_col, border_col) in CIPDashboard.PHASE_COLORS.items():
        mask   = state_arr == int(phase)
        if not any(mask):
            continue
        regions = _find_regions(t, mask)
        for (t0, t1) in regions:
            for ax in axes:
                ax.axvspan(t0, t1, alpha=0.15, color=border_col,
                           label=STATE_NAMES[phase] if ax is axes[0] else '')

    # ── Temperature ───────────────────────────────────────
    axes[0].plot(t, sim.hist_temp_pv, '#4de8ff', lw=1.5, label='Temp PV')
    axes[0].plot(t, sim.hist_temp_sp, '#ff6b6b', lw=1.0, ls='--',
                  label='Temp SP')
    axes[0].set_ylabel('Temperature [°C]', fontsize=8)
    axes[0].axhline(60.0, color='#ff3b3b', lw=0.8, ls=':', label='Trip 60°C')
    axes[0].legend(loc='upper right', fontsize=7,
                    facecolor='#0f1318', edgecolor='#2a3547')
    axes[0].set_ylim(10, 70)

    # ── Steam valve + PID output ──────────────────────────
    ax_s = axes[0].twinx()
    ax_s.plot(t, sim.hist_steam, '#ff9a6b', lw=1.0, alpha=0.6,
               label='Steam %')
    ax_s.set_ylabel('Steam Valve [%]', fontsize=7, color='#ff9a6b')
    ax_s.set_ylim(-5, 115)
    ax_s.tick_params(colors='#ff9a6b', labelsize=7)

    # ── Conductivity ──────────────────────────────────────
    axes[1].plot(t, sim.hist_cond, '#ffc845', lw=1.5, label='Cond PV')
    axes[1].axhline(sim.recipe.cond_gate, color='#ff3b3b', lw=1.0,
                     ls='--', label=f'Gate {sim.recipe.cond_gate} mS/cm')
    axes[1].set_ylabel('Conductivity [mS/cm]', fontsize=8)
    axes[1].legend(loc='upper right', fontsize=7,
                    facecolor='#0f1318', edgecolor='#2a3547')
    axes[1].set_ylim(0, max(sim.hist_cond) * 1.1 if sim.hist_cond else 15)

    # ── Flow ──────────────────────────────────────────────
    axes[2].plot(t, sim.hist_flow, '#69ff9a', lw=1.5, label='Flow')
    axes[2].axhline(5.0, color='#ff6b6b', lw=0.8, ls='--',
                     label='Min permissive')
    axes[2].set_ylabel('Flow [L/min]', fontsize=8)
    axes[2].legend(loc='upper right', fontsize=7,
                    facecolor='#0f1318', edgecolor='#2a3547')
    axes[2].set_ylim(-2, 35)

    # ── Chemical consumption ──────────────────────────────
    axes[3].plot(t, sim.hist_naoh_vol, '#ffc845', lw=1.5, label='NaOH [L]')
    axes[3].plot(t, sim.hist_acac_vol, '#ff6b6b', lw=1.5, label='AcAc [L]')
    axes[3].set_ylabel('Chemical [L]', fontsize=8)
    axes[3].set_xlabel('Time [min]', fontsize=8)
    axes[3].legend(loc='upper left', fontsize=7,
                    facecolor='#0f1318', edgecolor='#2a3547')

    # ── Add phase transition markers ──────────────────────
    if sim.state_machine.transition_log:
        for (tt, s_from, s_to) in sim.state_machine.transition_log:
            if s_to in CIPDashboard.PHASE_COLORS:
                for ax in axes:
                    ax.axvline(tt / 60.0, color='#2a3547',
                                lw=0.6, ls=':')
                axes[0].text(tt/60.0 + 0.3, 62,
                              STATE_NAMES[s_to][:8],
                              fontsize=6, color='#5a6a80', rotation=90,
                              va='bottom')

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig('CIP_Batch_Report.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0c10')
    print(f"\n  Plot saved: CIP_Batch_Report.png")
    plt.show()


def _find_regions(t_arr, mask):
    """Find contiguous True regions in mask, return list of (t0, t1)."""
    regions = []
    in_region = False
    t0 = 0.0
    for i, (ti, m) in enumerate(zip(t_arr, mask)):
        if m and not in_region:
            t0 = ti
            in_region = True
        elif not m and in_region:
            regions.append((t0, ti))
            in_region = False
    if in_region:
        regions.append((t0, t_arr[-1]))
    return regions


# ═══════════════════════════════════════════════════════════════════════
# SECTION 10 — OPC UA HARDWARE-IN-THE-LOOP (optional)
# ═══════════════════════════════════════════════════════════════════════

def run_opcua_hil(recipe_id: int):
    """
    Hardware-in-the-loop: Python reads sensor values from CODESYS
    OPC UA server, runs the simulation, and writes simulated PVs
    back to CODESYS AI inputs for use in SoftPLC testing.

    REQUIRES: pip install asyncua
    REQUIRES: CODESYS running with OPC UA server enabled on port 4840
    """
    try:
        import asyncio
        from asyncua import Client, ua
    except ImportError:
        print("\n  ERROR: asyncua not installed.")
        print("  Install with: pip install asyncua")
        return

    SERVER_URL = "opc.tcp://localhost:4840"

    # OPC UA node IDs — match these to your CODESYS Symbol Configuration
    # Format: ns=4;s=|var|Application.GVL_IO.r_Temp_PV
    NODE_TEMP_PV   = "ns=4;s=|var|Application.GVL_IO.r_Temp_PV"
    NODE_COND_PV   = "ns=4;s=|var|Application.GVL_IO.r_Cond_PV"
    NODE_FLOW_PV   = "ns=4;s=|var|Application.GVL_IO.r_Flow_PV"
    NODE_TEMP_AI   = "ns=4;s=|var|Application.GVL_IO.w_Temp_RawAI"
    NODE_COND_AI   = "ns=4;s=|var|Application.GVL_IO.w_Cond_RawAI"
    NODE_STATE     = "ns=4;s=|var|Application.GVL_Alarms.n_CIP_State"
    NODE_SIM_EN    = "ns=4;s=|var|Application.GVL_IO.b_Sim_Enable"

    async def main():
        sim = CIPSimulation(recipe_id=recipe_id)
        print(f"\n  Connecting to CODESYS OPC UA at {SERVER_URL}...")

        async with Client(url=SERVER_URL) as client:
            print("  Connected. Starting hardware-in-the-loop simulation.")
            print("  Press Ctrl+C to stop.\n")

            # Enable simulation mode in CODESYS
            node_sim = client.get_node(NODE_SIM_EN)
            await node_sim.write_value(ua.DataValue(ua.Variant(True, ua.VariantType.Boolean)))

            loop_count = 0
            while True:
                # 1. Read PLC state for diagnostics
                try:
                    state_val = await client.get_node(NODE_STATE).read_value()
                    print(f"  PLC State: {state_val}  |  "
                          f"Sim Temp: {sim.temp_pv:.1f}°C  |  "
                          f"Sim Cond: {sim.cond_pv:.3f} mS/cm", end='\r')
                except Exception:
                    pass

                # 2. Run simulation step
                sim.step()
                if loop_count == 50:
                    sim.state_machine.cmd_start = True

                # 3. Write simulated values to CODESYS AI inputs (raw counts)
                temp_raw  = int(min(32767, max(0,
                             sim.temp_pv / 100.0 * 32767.0)))
                cond_raw  = int(min(32767, max(0,
                             sim.cond_pv / 20.0  * 32767.0)))

                await client.get_node(NODE_TEMP_AI).write_value(
                    ua.DataValue(ua.Variant(temp_raw, ua.VariantType.UInt16)))
                await client.get_node(NODE_COND_AI).write_value(
                    ua.DataValue(ua.Variant(cond_raw, ua.VariantType.UInt16)))

                loop_count += 1
                await asyncio.sleep(sim.dt)

    asyncio.run(main())


# ═══════════════════════════════════════════════════════════════════════
# SECTION 11 — LIVE ANIMATION MODE
# ═══════════════════════════════════════════════════════════════════════

def run_live_animation(recipe_id: int, inject_disturbance: bool):
    """Run the live animated dashboard."""
    sim  = CIPSimulation(recipe_id=recipe_id,
                          inject_disturbance=inject_disturbance)
    dash = CIPDashboard(sim)

    print(f"\n{'='*70}")
    print(f"  CIP DIGITAL TWIN — LIVE ANIMATION")
    print(f"  Recipe: {RECIPES[recipe_id].name}")
    print(f"  Simulation speed: {sim.sim_speed}x real-time")
    print(f"  Disturbances: {'ON' if inject_disturbance else 'OFF'}")
    print(f"  Close the plot window to stop.")
    print(f"{'='*70}\n")

    ani = FuncAnimation(
        dash.fig,
        dash.update,
        interval=100,   # 100ms refresh = 10 fps
        blit=False,
        cache_frame_data=False
    )

    plt.show()

    # Print batch report after window closed
    report = sim.get_batch_report()
    if report:
        print(f"\n{'─'*50}")
        print(f"  BATCH REPORT — {report.get('batch_id', '')}")
        print(f"{'─'*50}")
        for k, v in report.items():
            if isinstance(v, float):
                print(f"  {k:<30} {v:.3f}")
            else:
                print(f"  {k:<30} {v}")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='CIP Boiler Descaling — Python Digital Twin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python CIP_Digital_Twin.py                    Run live animation, recipe 1
  python CIP_Digital_Twin.py --static           Run headless, print report + PNG
  python CIP_Digital_Twin.py --recipe 2         Use HEAVY-SCALE recipe
  python CIP_Digital_Twin.py --disturbance      Inject process disturbances
  python CIP_Digital_Twin.py --opcua            OPC UA hardware-in-the-loop
  python CIP_Digital_Twin.py --static --recipe 4 --disturbance  Combined
        """
    )
    parser.add_argument('--recipe',      type=int, default=1, choices=[1,2,3,4],
                         help='Recipe number 1-4 (default: 1)')
    parser.add_argument('--static',      action='store_true',
                         help='Run headless full-batch simulation')
    parser.add_argument('--disturbance', action='store_true',
                         help='Inject process disturbances')
    parser.add_argument('--opcua',       action='store_true',
                         help='OPC UA hardware-in-the-loop mode')
    parser.add_argument('--speed',       type=float, default=60.0,
                         help='Animation speed multiplier (default: 60)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║       CIP BOILER DESCALING — PYTHON DIGITAL TWIN v2.0          ║
  ╚══════════════════════════════════════════════════════════════════╝
    """)

    if args.opcua:
        run_opcua_hil(args.recipe)
    elif args.static:
        run_static_simulation(args.recipe, args.disturbance)
    else:
        sim_live = None
        try:
            import matplotlib
            matplotlib.use('TkAgg')
        except Exception:
            pass
        run_live_animation(args.recipe, args.disturbance)
