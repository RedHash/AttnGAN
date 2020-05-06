import torch
import torch.nn as nn

from miscc.config import cfg


class FeaturesGenerator(nn.Module):
    def __init__(self, text_dim=11083, X_dim=3584):
        super(FeaturesGenerator, self).__init__()
        self.rdc_text = nn.Linear(text_dim, cfg.FEATURE_SYNTHESIS.H_TEXT_DIM)

        self.main = nn.Sequential(nn.Linear(cfg.FEATURE_SYNTHESIS.Z_DIM + cfg.FEATURE_SYNTHESIS.H_TEXT_DIM,
                                            cfg.FEATURE_SYNTHESIS.H_DIM),
                                  nn.LeakyReLU(),
                                  nn.Linear(cfg.FEATURE_SYNTHESIS.H_DIM, X_dim),
                                  nn.Tanh())

    def forward(self, z, c):
        rdc_text = self.rdc_text(c)
        input = torch.cat([z, rdc_text], 1)
        output = self.main(input)
        return output
