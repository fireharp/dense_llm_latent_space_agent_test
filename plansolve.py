"""PlanSolve: DSPy module with dense latent communication."""

import dspy
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from dense_lm import DenseLM
from dense_edge import DenseEdge


class DensePlanSolve(dspy.Module):
    """DSPy module that chains Planner → Edge → Solver using hidden states.
    
    This module implements a two-stage problem solving approach where:
    1. Planner encodes the problem into hidden states
    2. Edge transforms the hidden states
    3. Solver decodes the transformed states into a solution
    """
    
    def __init__(
        self,
        planner_lm: Optional[DenseLM] = None,
        solver_lm: Optional[DenseLM] = None,
        edge: Optional[DenseEdge] = None,
        device: Optional[str] = None,
        share_lm: bool = True,
    ):
        """Initialize PlanSolve module.
        
        Args:
            planner_lm: DenseLM instance for planning
            solver_lm: DenseLM instance for solving (can be same as planner)
            edge: DenseEdge transformer for inter-module communication
            device: Device to run on
            share_lm: Whether to share the same LM for planner and solver
        """
        super().__init__()
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Initialize LMs
        if planner_lm is None:
            self.planner = DenseLM(device=device)
        else:
            self.planner = planner_lm
            
        if share_lm:
            self.solver = self.planner
        else:
            if solver_lm is None:
                self.solver = DenseLM(device=device)
            else:
                self.solver = solver_lm
                
        # Initialize edge
        if edge is None:
            self.edge = DenseEdge(d_model=self.planner.hidden_size).to(device)
        else:
            self.edge = edge.to(device)
            
        # DSPy signatures for planning and solving
        self.plan_signature = dspy.Signature("goal -> plan")
        self.solve_signature = dspy.Signature("problem, plan -> solution")
        
    def forward(self, goal: str, use_dense: bool = True) -> Dict[str, Any]:
        """Execute the plan-solve pipeline.
        
        Args:
            goal: The problem or task to solve
            use_dense: Whether to use dense communication (vs text)
            
        Returns:
            Dictionary containing solution and intermediate outputs
        """
        if use_dense:
            return self._forward_dense(goal)
        else:
            return self._forward_text(goal)
            
    def _forward_dense(self, goal: str) -> Dict[str, Any]:
        """Execute using dense hidden state communication."""
        # Step 1: Encode goal into hidden states using planner
        planning_prompt = f"Problem: {goal}\nLet me create a step-by-step plan:"
        h_plan = self.planner.encode(planning_prompt)
        
        # Step 2: Transform hidden states through edge
        h_transformed = self.edge(h_plan)
        
        # Step 3: Decode solution from transformed hidden states
        solution = self.solver.decode(h_transformed)
        
        # Also generate a plan for interpretability (optional)
        with dspy.context(lm=self.planner):
            plan_output = dspy.Predict(self.plan_signature)(goal=goal)
            plan = plan_output.plan
        
        return {
            "solution": solution,
            "plan": plan,
            "hidden_states": {
                "h_plan": h_plan,
                "h_transformed": h_transformed,
            },
            "mode": "dense"
        }
        
    def _forward_text(self, goal: str) -> Dict[str, Any]:
        """Execute using traditional text-based communication."""
        # Step 1: Generate plan using planner
        with dspy.context(lm=self.planner):
            plan_output = dspy.Predict(self.plan_signature)(goal=goal)
            plan = plan_output.plan
            
        # Step 2: Solve using the plan
        with dspy.context(lm=self.solver):
            solve_output = dspy.Predict(self.solve_signature)(
                problem=goal,
                plan=plan
            )
            solution = solve_output.solution
            
        return {
            "solution": solution,
            "plan": plan,
            "mode": "text"
        }
        
    def get_trainable_parameters(self):
        """Get trainable parameters (edge only by default)."""
        return self.edge.parameters()
        
    def freeze_lms(self):
        """Freeze language model parameters."""
        # Freeze planner
        for param in self.planner.model.parameters():
            param.requires_grad = False
            
        # Freeze solver if different from planner
        if self.solver is not self.planner:
            for param in self.solver.model.parameters():
                param.requires_grad = False
                
    def unfreeze_lms(self):
        """Unfreeze language model parameters."""
        # Unfreeze planner
        for param in self.planner.model.parameters():
            param.requires_grad = True
            
        # Unfreeze solver if different from planner
        if self.solver is not self.planner:
            for param in self.solver.model.parameters():
                param.requires_grad = True
                
    def save_edge_weights(self, path: str):
        """Save edge weights."""
        self.edge.save_state_dict(path)
        
    def load_edge_weights(self, path: str):
        """Load edge weights."""
        self.edge.load_state_dict_from_path(path)


class PromptPlanSolve(dspy.Module):
    """Baseline text-based PlanSolve module for comparison."""
    
    def __init__(self, lm: Optional[dspy.LM] = None):
        """Initialize baseline PlanSolve.
        
        Args:
            lm: Language model to use (will create DenseLM if None)
        """
        super().__init__()
        
        if lm is None:
            self.lm = DenseLM()
        else:
            self.lm = lm
            
        # DSPy chain of thought for planning and solving
        self.plan_solve = dspy.ChainOfThought("goal -> plan, solution")
        
    def forward(self, goal: str) -> Dict[str, Any]:
        """Execute the plan-solve pipeline using text.
        
        Args:
            goal: The problem or task to solve
            
        Returns:
            Dictionary containing solution and plan
        """
        with dspy.context(lm=self.lm):
            output = self.plan_solve(goal=goal)
            
        return {
            "solution": output.solution,
            "plan": output.plan,
            "mode": "text_baseline",
            "completions": output.completions if hasattr(output, 'completions') else None
        }