import streamlit as st
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mesa
# --- Try alternative import for older Mesa versions ---
try:
    from mesa.time import RandomActivation
except ModuleNotFoundError:
    try:
        # Fallback for potentially older Mesa structure
        from mesa import RandomActivation
    except ImportError:
        st.error("Could not import RandomActivation scheduler from Mesa. Please ensure Mesa is installed correctly in your environment (`pip install mesa`).")
        st.stop() # Stop execution if scheduler can't be imported

# --- Agent Definitions (Mesa Agents) ---

class JudgeAgent(mesa.Agent):
    """Represents a single Supreme Court Justice."""
    def __init__(self, unique_id, model, philosophy, collegial_influence):
        super().__init__(unique_id, model)
        self.philosophy = philosophy  # 1 (Formalist) to 10 (Activist)
        self.collegial_influence = collegial_influence # How easily swayed (0 to 1)
        self.current_vote = None # To store vote during deliberation

    def initial_vote(self, law_infringement_score, is_basic_law_attack=False):
        """Determines the judge's initial vote based on philosophy."""
        # Special logic for existential threats (like basic law attacks)
        if is_basic_law_attack:
             # Even formalists may vote to invalidate if core power is attacked
             threshold = 3 # Lower threshold significantly for existential threats
        else:
            threshold = 11 - self.philosophy # Normal threshold

        if law_infringement_score >= threshold:
            self.current_vote = "Invalidate"
        else:
            self.current_vote = "Uphold"
        return self.current_vote

    def deliberate_and_final_vote(self, panel_initial_votes, global_legitimacy, is_basic_law_attack=False):
        """Determines the final vote after influence and legitimacy check."""
        invalidate_count = panel_initial_votes.count("Invalidate")
        uphold_count = len(panel_initial_votes) - invalidate_count
        final_decision = self.current_vote

        flip_probability_modifier = (1 - self.collegial_influence)
        # Simplified influence model
        if self.current_vote == "Invalidate" and uphold_count >= len(panel_initial_votes) * 0.7:
             if random.random() < (0.5 * flip_probability_modifier):
                 final_decision = "Uphold"
        elif self.current_vote == "Uphold" and invalidate_count >= len(panel_initial_votes) * 0.7:
             if random.random() < (0.5 * flip_probability_modifier):
                 final_decision = "Invalidate"

        # Legitimacy Check (less effective for existential threats)
        legitimacy_threshold = 50
        deference_increase_factor = 2.0
        if not is_basic_law_attack: # Legitimacy check less relevant for core attacks
            if global_legitimacy < legitimacy_threshold and final_decision == "Invalidate":
                flip_probability = ((legitimacy_threshold - global_legitimacy) / legitimacy_threshold) * deference_increase_factor
                if random.random() < np.clip(flip_probability, 0, 1):
                    final_decision = "Uphold"

        self.current_vote = final_decision
        return final_decision

class MKAgent(mesa.Agent):
    """Represents a single Member of Knesset."""
    def __init__(self, unique_id, model, is_coalition, ideology, party_discipline, electoral_security):
        super().__init__(unique_id, model)
        self.is_coalition = is_coalition
        self.ideology = ideology
        self.party_discipline = party_discipline
        self.electoral_security = electoral_security
        self.hostility = 0 # 0 to 10

    def vote_on_law(self, coalition_position_is_for, law_infringement_score):
        """Determines the MK's vote (1 for 'yes', 0 for 'no')."""
        if self.is_coalition:
            if random.random() < self.party_discipline:
                return 1 if coalition_position_is_for else 0
            else: # Discipline failed, check ideology
                ideological_conflict = False
                # Simplified check: Only defect if ideology strongly conflicts with law type
                if coalition_position_is_for and self.ideology < 4 and law_infringement_score > 6: # Left MK vs High Infringe
                    ideological_conflict = True
                elif coalition_position_is_for and self.ideology > 6 and law_infringement_score < 4: # Right MK vs Low Infringe (less likely)
                     ideological_conflict = (random.random() < 0.1) # Only small chance even if discipline fails

                if ideological_conflict:
                    return 0 if coalition_position_is_for else 1 # Defect
                else:
                    return 1 if coalition_position_is_for else 0 # No conflict or discipline holds
        else: # Opposition
            return 0 if coalition_position_is_for else 1 # Always vote against coalition

    def vote_on_override(self, law_infringement_score):
        """Determines if a coalition MK votes for an override (1 yes, 0 no)."""
        if not self.is_coalition:
            return 0

        base_propensity = (self.party_discipline * 0.7 + self.hostility / 10 * 0.3)
        self_restraint_factor = 1.0
        if 4 <= self.ideology <= 7: # Moderate MKs
             extreme_infringement_threshold = 9.0
             if law_infringement_score >= extreme_infringement_threshold:
                 self_restraint_factor = 0.8 # Slight hesitation for extreme laws
        override_propensity = np.clip(base_propensity * self_restraint_factor, 0, 1)
        return 1 if random.random() < override_propensity else 0

    def update_hostility(self, legitimacy_hit):
        increase = abs(legitimacy_hit) / 4
        self.hostility = min(10, self.hostility + increase)

    def decrease_hostility(self):
        self.hostility = max(0, self.hostility - 1.0)

# --- Mesa Model Definition ---

def get_legitimacy(model): return model.public_legitimacy
def get_avg_hostility(model):
    coalition_mks = [a for a in model.schedule.agents if isinstance(a, MKAgent) and a.is_coalition]
    if not coalition_mks: return 0
    return np.mean([mk.hostility for mk in coalition_mks])
def get_desire_for_override(model):
    if hasattr(model, 'override_rule_active') and model.override_rule_active: return 0
    if hasattr(model, 'desire_for_override'): return model.desire_for_override
    return 0


class ConstitutionalModel(mesa.Model):
    """
    The main Mesa model for simulating Knesset-Court interactions.

    Methodological Note on Parameters:
    Conceptual model using archetypes. Legitimacy=65 reflects polls (e.g., IDI Index).
    Philosophy scores (9,6,3) model activist/moderate/formalist judges.
    Infringement=8 simulates a hard case. Future research could use empirical data
    (content analysis, polling, expert surveys).
    """
    def __init__(self, scenario):
        super().__init__()
        self.scenario_name = scenario
        self.schedule = RandomActivation(self)
        self.public_legitimacy = 65.0
        self.override_rule_active = False
        self.desire_for_override = 0
        self.running = True
        self.laws_passed_this_turn = 0
        self.laws_invalidated_this_turn = 0
        self.overrides_this_turn = 0
        self.current_log_entry = ""
        self.institutional_challenge_score = 0 # For scenario F/G

        self._initialize_model_parameters(scenario)
        self._create_agents()

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Legitimacy": get_legitimacy,
                "Avg Coalition Hostility": get_avg_hostility,
                "Desire for Override": get_desire_for_override
            }
        )
        self.datacollector.collect(self)

    def _initialize_model_parameters(self, scenario):
         # Defaults
         self.num_judges = NUM_JUDGES; self.num_mks = NUM_MKS; self.coalition_size = COALITION_SIZE
         self.panel_size = PANEL_SIZE; self.override_threshold = OVERRIDE_THRESHOLD
         # Reset flags
         self.bergman_entrenched_rule_active = False; self.force_enlistment_law = False
         self.override_is_limited = False; self.crisis_active = False
         self.override_rule_active = False; self.desire_for_override = 0
         self.attack_reasonableness = False # Flag for Scenario F
         self.modern_bergman_test = False # Flag for Scenario G

         if scenario == "Status Quo (Scenario A)":
            self.avg_philosophy = 5.5; self.std_dev_philosophy = 2.0
            self.public_legitimacy = 65.0; self.avg_ideology = 7.5; self.avg_discipline = 0.9
         elif scenario == "Bergman Era (Scenario B)":
            self.avg_philosophy = 2.0; self.std_dev_philosophy = 0.5
            self.public_legitimacy = 80.0; self.avg_ideology = 6.0; self.avg_discipline = 0.95
            self.bergman_entrenched_rule_active = True
         elif scenario == "Override Active (Scenario C)":
            self.avg_philosophy = 5.5; self.std_dev_philosophy = 2.0
            self.public_legitimacy = 65.0; self.override_rule_active = True
            self.avg_ideology = 7.5; self.avg_discipline = 0.9
         elif scenario == "Enlistment Dynamic (Scenario D)":
            self.avg_philosophy = 5.5; self.std_dev_philosophy = 2.0
            self.public_legitimacy = 65.0; self.avg_ideology = 7.5; self.avg_discipline = 0.9
            self.force_enlistment_law = True; self.desire_for_override = 0
         elif scenario == "1994 Compromise (Scenario E)":
             self.avg_philosophy = 5.0; self.std_dev_philosophy = 1.5
             self.public_legitimacy = 70.0; self.override_rule_active = True
             self.override_is_limited = True; self.avg_ideology = 7.0; self.avg_discipline = 0.85
             self.crisis_active = True
         # --- NEW SCENARIOS ---
         elif scenario == "Reasonableness Attack (Scenario F)":
            self.avg_philosophy = 5.5; self.std_dev_philosophy = 2.0
            self.public_legitimacy = 65.0; self.avg_ideology = 8.0 # More right-wing coalition assumed
            self.avg_discipline = 0.9
            self.attack_reasonableness = True # Flag to trigger the specific legislative action
         elif scenario == "Modern Bergman Test (Scenario G)":
            self.avg_philosophy = 5.5; self.std_dev_philosophy = 2.0
            self.public_legitimacy = 65.0; self.avg_ideology = 7.5; self.avg_discipline = 0.9
            self.modern_bergman_test = True # Flag to trigger the specific legislative action
         # --- END NEW SCENARIOS ---
         else: # Default Status Quo
            self.avg_philosophy = 5.5; self.std_dev_philosophy = 2.0
            self.public_legitimacy = 65.0; self.avg_ideology = 7.5; self.avg_discipline = 0.9

    def _create_agents(self):
        # ... (Agent creation code remains the same) ...
        self.schedule = RandomActivation(self)
        for i in range(self.num_judges):
            philosophy = np.clip(random.gauss(self.avg_philosophy, self.std_dev_philosophy), 1, 10)
            influence = random.uniform(0.1, 0.9)
            self.schedule.add(JudgeAgent(f"J_{i}", self, philosophy, influence))
        for i in range(self.num_mks):
            is_coalition = i < self.coalition_size
            if is_coalition: ideology = np.clip(random.gauss(self.avg_ideology, 1.5), 1, 10)
            else: ideology = np.clip(random.gauss(max(1, self.avg_ideology - 3), 2.0), 1, 10)
            discipline = np.clip(random.gauss(self.avg_discipline, 0.05), 0.8, 1.0)
            security = random.random()
            self.schedule.add(MKAgent(f"M_{i}", self, is_coalition, ideology, discipline, security))


    def get_mks(self): return [a for a in self.schedule.agents if isinstance(a, MKAgent)]
    def get_judges(self): return [a for a in self.schedule.agents if isinstance(a, JudgeAgent)]

    def step(self):
        self.laws_passed_this_turn = 0; self.laws_invalidated_this_turn = 0
        self.overrides_this_turn = 0; self.current_log_entry = f"**Turn {self.schedule.steps + 1}:** "

        # --- Step 1: Knesset Legislates ---
        mks = self.get_mks(); coalition_mks = [mk for mk in mks if mk.is_coalition]
        if not coalition_mks: self.current_log_entry += "No coalition MKs..."; self.schedule.step(); self.datacollector.collect(self); return

        infringement_score = 0
        is_basic_law_attack = False # Flag for judicial response
        is_modern_bergman_violation = False # Flag for judicial response

        # --- Scenario Specific Legislation ---
        if hasattr(self, 'force_enlistment_law') and self.force_enlistment_law:
            infringement_score = 9.0; self.current_log_entry += f"Coalition forced: Enlistment Law (Infringe: {infringement_score:.1f}). "
        elif hasattr(self, 'attack_reasonableness') and self.attack_reasonableness and self.schedule.steps == 0: # Only propose in first turn for simplicity
            infringement_score = 10.0 # Represents institutional attack, not rights infringement per se
            is_basic_law_attack = True
            self.current_log_entry += f"Coalition proposes Basic Law Amendment to abolish Reasonableness (Institutional Attack: {infringement_score:.1f}). "
        elif hasattr(self, 'modern_bergman_test') and self.modern_bergman_test and self.schedule.steps == 0:
            infringement_score = 6.0 # Moderate infringement, but violates entrenched rule
            is_modern_bergman_violation = True
            self.current_log_entry += f"Coalition proposes law violating entrenched rule (Infringe: {infringement_score:.1f}, Bergman Test). "
        # --- End Scenario Specific Legislation ---
        else: # Default legislation proposal
            avg_coalition_ideology = np.mean([mk.ideology for mk in coalition_mks]); infringement_score = np.clip(random.gauss(avg_coalition_ideology * 0.8, 2.5), 1, 10); self.current_log_entry += f"Knesset proposes law (Infringe: {infringement_score:.1f}). "

        coalition_votes_for = sum(mk.vote_on_law(True, infringement_score) for mk in coalition_mks)
        required_majority = 61; law_passed = coalition_votes_for >= required_majority
        opposition_mks = [mk for mk in mks if not mk.is_coalition]; opposition_votes_against = sum(1 - mk.vote_on_law(True, infringement_score) for mk in opposition_mks)
        coalition_votes_against = self.coalition_size - coalition_votes_for; total_votes_against = opposition_votes_against + coalition_votes_against

        if not law_passed: self.current_log_entry += f"Law FAILED Knesset ({coalition_votes_for} For / {total_votes_against} Against). "; self.schedule.step(); self.datacollector.collect(self); return
        self.laws_passed_this_turn += 1; self.current_log_entry += f"Law PASSED Knesset ({coalition_votes_for} For / {total_votes_against} Against). "

        # --- Step 2: Court Deliberates ---
        judicial_review_threshold = 3; is_bergman_violation = False
        if hasattr(self, 'bergman_entrenched_rule_active') and self.bergman_entrenched_rule_active and infringement_score > 7: is_bergman_violation = True; self.current_log_entry += "[Bergman Rule Triggered]. "

        # --- Determine if review happens ---
        needs_review = (infringement_score >= judicial_review_threshold or is_bergman_violation or is_basic_law_attack or is_modern_bergman_violation)
        if not needs_review:
             self.current_log_entry += "Law not reviewed. "; self.schedule.step(); self.datacollector.collect(self); return

        judges = self.get_judges();
        if len(judges) < self.panel_size: self.current_log_entry += "Not enough judges..."; self.schedule.step(); self.datacollector.collect(self); return
        panel = random.sample(judges, self.panel_size)

        original_philosophies = {};
        if is_bergman_violation or is_modern_bergman_violation: # Apply boost for clear rule violation
            for judge in panel: original_philosophies[judge.unique_id] = judge.philosophy; judge.philosophy = 8.0

        # Pass flag for basic law attack to voting methods
        initial_votes = [j.initial_vote(infringement_score, is_basic_law_attack) for j in panel];
        final_votes_list = [j.deliberate_and_final_vote(initial_votes, self.public_legitimacy, is_basic_law_attack) for j in panel]

        if is_bergman_violation or is_modern_bergman_violation: # Restore philosophies
            for judge in panel:
                if judge.unique_id in original_philosophies: judge.philosophy = original_philosophies[judge.unique_id]

        invalidate_count = final_votes_list.count("Invalidate"); uphold_count = self.panel_size - invalidate_count
        court_decision = "Invalidate" if invalidate_count > uphold_count else "Uphold"
        self.current_log_entry += f"Court rules: {court_decision} ({invalidate_count}-{uphold_count}). "

        # --- Step 3: System Reacts ---
        legitimacy_change = 0
        if court_decision == "Invalidate":
            self.laws_invalidated_this_turn += 1
            # Apply bigger legitimacy hit for invalidating a Basic Law attack
            legitimacy_hit_multiplier = 2.5 if is_basic_law_attack else 1.5
            legitimacy_change = - (infringement_score * legitimacy_hit_multiplier) # Bigger hit
            self.current_log_entry += f"Legitimacy decreases by {abs(legitimacy_change):.1f}. "
            for mk in coalition_mks: mk.update_hostility(legitimacy_change) # Hostility increases more too

            # Increase Desire for Override ONLY if override NOT active
            if not self.override_rule_active:
                is_modern_scenario = not (hasattr(self, 'bergman_entrenched_rule_active') and self.bergman_entrenched_rule_active) and \
                                     not (hasattr(self, 'override_is_limited') and self.override_is_limited)
                if is_modern_scenario:
                     increase_amount = 15 if (hasattr(self, 'force_enlistment_law') and self.force_enlistment_law) else 5
                     # Bigger jump if basic law invalidated
                     increase_amount = increase_amount * 2 if is_basic_law_attack else increase_amount
                     self.desire_for_override = min(100, self.desire_for_override + increase_amount)
                     self.current_log_entry += f"[Desire for Override: {self.desire_for_override}]. "

            override_successful = False
            if self.override_rule_active:
                # ... (override logic remains the same) ...
                is_limited_override = hasattr(self, 'override_is_limited') and self.override_is_limited; can_override_this_law = True
                if is_limited_override and infringement_score >= 8: can_override_this_law = False; self.current_log_entry += "[Limited Override Not Applicable]. "

                if can_override_this_law:
                    override_votes = sum(mk.vote_on_override(infringement_score) for mk in coalition_mks)
                    if override_votes >= self.override_threshold:
                        override_successful = True; self.overrides_this_turn += 1; self.laws_invalidated_this_turn = 0
                        self.current_log_entry += f"Knesset OVERRIDES ({override_votes} votes). Law stands. "
                        if not is_limited_override: legitimacy_change -= 15; self.current_log_entry += f"Additional legitimacy hit (-15). "
                        else: legitimacy_change += 5; self.current_log_entry += "Legitimacy stabilized by compromise override (+5)."
                    else:
                         self.current_log_entry += f"Override vote FAILED ({override_votes} votes). Law invalidated. "
                         if infringement_score >= 9.0: self.current_log_entry += "(Self-restraint check possibly active for moderates). "


            # Bergman compliance logic
            if not override_successful and (is_bergman_violation or is_modern_bergman_violation):
                 # Compliance less certain in modern era
                 if hasattr(self, 'bergman_entrenched_rule_active') and self.bergman_entrenched_rule_active:
                     self.current_log_entry += "Knesset complies (Bergman dynamic). "; legitimacy_change += 5
                 else: # Modern Bergman
                     # Calculate average hostility
                     avg_hostility_now = np.mean([mk.hostility for mk in coalition_mks]) if coalition_mks else 0
                     compliance_probability = max(0.1, 1 - (avg_hostility_now / 10)) # Higher hostility = lower compliance chance
                     if random.random() < compliance_probability:
                         self.current_log_entry += "Knesset complies (Modern Bergman). "; legitimacy_change += 2 # Smaller boost
                     else:
                         self.current_log_entry += "Knesset DEFIES! (Modern Bergman). " # No legitimacy boost, hostility remains high


        else: # Court Upholds
            legitimacy_change = infringement_score / 3; self.current_log_entry += f"Legitimacy increases slightly by {legitimacy_change:.1f}. "
            for mk in coalition_mks: mk.decrease_hostility()
            # Decay Desire for Override ONLY if override NOT active
            if not self.override_rule_active:
                is_modern_scenario = not (hasattr(self, 'bergman_entrenched_rule_active') and self.bergman_entrenched_rule_active) and \
                                     not (hasattr(self, 'override_is_limited') and self.override_is_limited)
                if is_modern_scenario and hasattr(self, 'desire_for_override'):
                    self.desire_for_override = max(0, self.desire_for_override - 1.0)
                    self.current_log_entry += f"[Desire for Override decays to: {self.desire_for_override}]. "

            if hasattr(self, 'crisis_active') and self.crisis_active: self.crisis_active = False; self.current_log_entry += "[1994 Crisis Resolved]. "

        self.public_legitimacy = np.clip(self.public_legitimacy + legitimacy_change, 0, 100)
        self.current_log_entry += f"New Legitimacy: {self.public_legitimacy:.1f}."
        if hasattr(self, 'desire_for_override') and self.desire_for_override > 75: self.current_log_entry += " [Political Strategy Shift: Prioritize Override Clause!] "

        self.schedule.step()
        self.datacollector.collect(self)

# --- Constants ---
NUM_JUDGES = 15; NUM_MKS = 120; COALITION_SIZE = 62; PANEL_SIZE = 9; OVERRIDE_THRESHOLD = 61

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("🇮🇱 Constitutional ABM (Mesa): Knesset-Court Interaction")
st.markdown("Conceptual ABM using Mesa & Streamlit.")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Setup")
# --- Updated Scenario Options ---
scenario_options = [
    "Status Quo (Scenario A)",
    "Bergman Era (Scenario B)",
    "Override Active (Scenario C)",
    "Enlistment Dynamic (Scenario D)",
    "1994 Compromise (Scenario E)",
    "Reasonableness Attack (Scenario F)", # Added
    "Modern Bergman Test (Scenario G)"   # Added
    ]
# --- End Update ---
scenario = st.sidebar.selectbox("Select Scenario:", scenario_options, key='main_scenario_selector')
if st.sidebar.button("Initialize Model for Scenario"): st.session_state.model = ConstitutionalModel(scenario); st.session_state.initialized = True; st.session_state.step_logs = []; st.rerun()

# --- Main Area Display ---
if 'initialized' in st.session_state and st.session_state.initialized:
    # ... (Rest of Streamlit UI code remains the same) ...
    model = st.session_state.model
    st.sidebar.header("Run Simulation")
    num_turns = st.sidebar.slider("Number of Turns to Run:", 1, 50, 5, key='num_turns_runner')
    if st.sidebar.button("Run Turns"):
        if 'step_logs' not in st.session_state: st.session_state.step_logs = []
        with st.spinner(f'Running {num_turns} turns...'):
            for i in range(num_turns):
                 if 'model' in st.session_state and st.session_state.model: st.session_state.model.step();
                 if hasattr(st.session_state.model, 'current_log_entry'): st.session_state.step_logs.append(st.session_state.model.current_log_entry)
                 else: st.warning("Model not found."); break
        st.rerun()
    if st.sidebar.button("Reset Simulation"): current_scenario = st.session_state.main_scenario_selector; st.session_state.model = ConstitutionalModel(current_scenario); st.session_state.step_logs = []; st.rerun()

    st.header(f"Current State (Turn {model.schedule.steps})")
    col1, col2 = st.columns(2);
    with col1:
        st.metric("Court Legitimacy", f"{model.public_legitimacy:.1f}/100"); st.subheader("Agents Overview")
        judges = model.get_judges(); mks = model.get_mks(); coalition_mks = [mk for mk in mks if mk.is_coalition]
        st.write(f"- Judges: {len(judges)} (Avg. Phil: {np.mean([j.philosophy for j in judges]):.1f})")
        st.write(f"- MKs: {len(mks)} ({len(coalition_mks)} Co / {model.num_mks-len(coalition_mks)} Op)")
        avg_hostility = np.mean([mk.hostility for mk in coalition_mks]) if coalition_mks else 0; avg_ideology = np.mean([mk.ideology for mk in coalition_mks]) if coalition_mks else 0
        st.write(f"- Coalition: Avg. Ideo: {avg_ideology:.1f}, Avg. Host: {avg_hostility:.1f}/10")
    with col2:
        st.subheader("Scenario Parameters"); st.write(f"**Scenario:** {model.scenario_name}")
        st.write(f"**Override Rule Active:** {'Yes' if model.override_rule_active else 'No'}")
        if hasattr(model, 'override_is_limited') and model.override_is_limited: st.write(f"**Override Type:** Limited (1994 Style)")

    st.header("Simulation Log"); log_key = f"log_display_{model.schedule.steps}"
    if 'step_logs' in st.session_state and st.session_state.step_logs:
         log_content = "\n\n".join(st.session_state.step_logs[-20:][::-1])
         st.markdown(f'<div style="height:300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; font-family: monospace;">{log_content.replace(" ", "&nbsp;").replace("/n","<br>")}</div>', unsafe_allow_html=True)
    else: st.write("Run simulation to see log entries.")

    st.header("Results Over Time")
    try:
        model_data = model.datacollector.get_model_vars_dataframe()
        if len(model_data) > 1:
            plot_col1, plot_col2, plot_col3 = st.columns(3) # Use 3 columns now
            with plot_col1: # Legitimacy Plot
                fig, ax = plt.subplots(); ax.plot(model_data.index, model_data["Legitimacy"], marker='o', linestyle='-'); ax.set_xlabel("Turn"); ax.set_ylabel("Legitimacy (0-100)"); ax.set_title("Court Legitimacy"); ax.set_ylim(0, 101); ax.grid(True); st.pyplot(fig)
            with plot_col2: # Hostility Plot
                fig2, ax2 = plt.subplots(); ax2.plot(model_data.index, model_data["Avg Coalition Hostility"], marker='x', linestyle='--', color='red'); ax2.set_xlabel("Turn"); ax2.set_ylabel("Hostility (0-10)"); ax2.set_title("Avg Coalition Hostility"); ax2.set_ylim(0, 10.1); ax2.grid(True); st.pyplot(fig2)
            # --- Desire for Override Plot ---
            with plot_col3:
                # Always try to plot, but handle cases where it's zero or non-existent
                if "Desire for Override" in model_data.columns and model_data["Desire for Override"].sum() > 0 : # Only plot if data exists and is non-zero
                    fig3, ax3 = plt.subplots(); ax3.plot(model_data.index, model_data["Desire for Override"], marker='^', linestyle=':', color='green'); ax3.set_xlabel("Turn"); ax3.set_ylabel("Desire (0-100)"); ax3.set_title("Desire for Override"); ax3.set_ylim(0, 101); ax3.grid(True); st.pyplot(fig3)
                else:
                    # Provide a placeholder or message if override is active or desire is zero
                     if model.override_rule_active:
                         st.write("(Override is active; desire irrelevant)")
                     else:
                         st.write("(Desire for Override remained at zero)")

        else: st.write("Run more turns to see results over time.")
    except Exception as e: st.error(f"Error plotting data: {e}"); st.write("Ensure model ran.")
else: st.info("Please initialize the model.")

