"""Tool intent heads for Bigram Tensor 1."""

import torch.nn as nn


TOOL_REGISTRY_DEFAULT = {
    0: "final",
    1: "web.search",
    2: "rag.search",
    3: "terminal.run",
    4: "python.run",
    5: "code.search",
    6: "citation.verify",
    7: "markdown.check",
    8: "calculator",
}


class ToolHead(nn.Module):
    """Predicts tool routing intent, tool name, and argument latent state."""

    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, 3)
        self.name = nn.Linear(config.hidden_size, config.n_tools)
        self.arg_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden):
        return {
            "tool_router_logits": self.router(hidden),
            "tool_name_logits": self.name(hidden),
            "tool_arg_hidden": self.arg_proj(hidden),
        }
