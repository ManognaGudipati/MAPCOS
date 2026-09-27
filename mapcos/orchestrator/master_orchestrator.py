"""
Master Orchestrator (LangGraph / CrewAI)
-------------------------------------------
Receives patient input and dispatches tasks to agents in parallel.

Implemented so far: CNN Classification Agent + Follicle Counting Agent run
concurrently on the ultrasound image, then their outputs are cross-checked
by the Grounding Agent.

Built to extend: lab_agent, symptoms_agent, synthesis_agent, and
exclusion_agent are accepted as optional constructor args. When they're
None (today), the orchestrator stops after grounding and returns that
result. Once you implement each one, pass it in and the corresponding
stage below activates automatically — no changes needed to the vision
agents, the grounding agent, or the code that dispatches them.
"""

from concurrent.futures import ThreadPoolExecutor


class MasterOrchestrator:
    def __init__(self, cnn_agent, follicle_agent, grounding_agent,
                 lab_agent=None, symptoms_agent=None,
                 synthesis_agent=None, exclusion_agent=None):
        self.cnn_agent = cnn_agent
        self.follicle_agent = follicle_agent
        self.grounding_agent = grounding_agent
        # Not yet implemented — orchestrator degrades gracefully when these are None.
        self.lab_agent = lab_agent
        self.symptoms_agent = symptoms_agent
        self.synthesis_agent = synthesis_agent
        self.exclusion_agent = exclusion_agent

    def _run_vision_agents(self, image_path: str) -> dict:
        with ThreadPoolExecutor(max_workers=2) as executor:
            cnn_future = executor.submit(self.cnn_agent.run, image_path)
            follicle_future = executor.submit(self.follicle_agent.run, image_path)
            return {
                "cnn": cnn_future.result(),
                "follicle": follicle_future.result(),
            }

    def _run_tabular_agents(self, tabular_data: dict) -> dict:
        results = {"lab": None, "symptoms": None}
        agents_to_run = {}
        if self.lab_agent is not None:
            agents_to_run["lab"] = (self.lab_agent.run, tabular_data.get("lab_values", {}))
        if self.symptoms_agent is not None:
            agents_to_run["symptoms"] = (self.symptoms_agent.run, tabular_data.get("symptoms", {}))

        if not agents_to_run:
            return results

        with ThreadPoolExecutor(max_workers=len(agents_to_run)) as executor:
            futures = {name: executor.submit(fn, arg) for name, (fn, arg) in agents_to_run.items()}
            for name, future in futures.items():
                results[name] = future.result()
        return results

    def run(self, patient_input: dict) -> dict:
        """
        patient_input:
            {
                "image_path": "<path to ultrasound image>",
                "tabular_data": {                     # optional, used once lab/symptoms agents exist
                    "lab_values": {...},
                    "symptoms": {...},
                }
            }
        """
        vision = self._run_vision_agents(patient_input["image_path"])
        grounding_result = self.grounding_agent.run(vision["cnn"], vision["follicle"])

        result = {
            "cnn_classification": vision["cnn"],
            "follicle_counting": vision["follicle"],
            "grounding": grounding_result,
            "lab": None,
            "symptoms": None,
            "synthesis": None,
            "exclusion": None,
        }

        tabular_data = patient_input.get("tabular_data")
        if tabular_data and (self.lab_agent or self.symptoms_agent):
            tabular = self._run_tabular_agents(tabular_data)
            result["lab"] = tabular["lab"]
            result["symptoms"] = tabular["symptoms"]

        # --- Not yet implemented: Diagnostic Synthesis Agent -------------------
        if self.synthesis_agent is not None:
            result["synthesis"] = self.synthesis_agent.run(
                grounding_result, result["lab"], result["symptoms"]
            )
            if self.exclusion_agent is not None:
                result["exclusion"] = self.exclusion_agent.run(result["synthesis"], result["lab"])
        # -------------------------------------------------------------------------

        return result
