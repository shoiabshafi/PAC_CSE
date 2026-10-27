"""PAC-CSE protocol constants."""

ARM_SIZE = 10
LAMBDA_X = 0.10
LAMBDA_P = 0.50
JEFFREYS = 0.5
VASE_SAMPLES = 10
VASE_ALPHA = 1.0
PAC_MAX_NEW_TOKENS = 16

PERTURBATIONS = (
    ("gauss_noise_s1", "gaussian noise", 0.08),
    ("gauss_noise_s2", "gaussian noise", 0.12),
    ("gauss_noise_s3", "gaussian noise", 0.18),
    ("gauss_blur_s1", "gaussian blur", 1),
    ("gauss_blur_s2", "gaussian blur", 2),
    ("gauss_blur_s3", "gaussian blur", 3),
    ("contrast_s1", "contrast", 0.4),
    ("contrast_s2", "contrast", 0.3),
    ("jpeg_s1", "jpeg", 25),
    ("pixelate_s2", "pixelation", 0.5),
)

MODEL_SAMPLING = {
    "Hulu-Med": (0.9, 0.9),
    "Lingshu": (0.9, 0.9),
    "CheXagent": (1.0, 0.95),
    "MedGemma": (1.0, 0.95),
}
