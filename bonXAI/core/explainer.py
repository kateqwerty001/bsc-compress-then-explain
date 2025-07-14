import shap
import sage


class SHAPExplainer:
    """
    Wrapper for SHAP explainers with kernel or permutation variants.
    """
    def __init__(self, model, variant: str = "kernel"):
        self.model = model
        self.variant = variant

    def explain(self, X, y=None):
        if self.variant == "kernel":
            explainer = shap.KernelExplainer(lambda x: self.model.predict_proba(x)[:, 1], X)
        elif self.variant == "permutation":
            masker = shap.maskers.Independent(X)
            explainer = shap.PermutationExplainer(lambda x: self.model.predict_proba(x)[:, 1], masker)
        else:
            raise ValueError(f"Unsupported SHAP variant: {self.variant}")
        
        shap_values = explainer(X)
        return shap_values.values


class SAGEExplainer:
    """
    Wrapper for SAGE explainers with permutation or kernel variants.
    """
    def __init__(self, model, variant: str = "permutation"):
        self.model = model
        self.variant = variant

    def explain(self, X, y):
        imputer = sage.MarginalImputer(self.model.predict_proba, X)

        if self.variant == "permutation":
            explainer = sage.PermutationEstimator(imputer, loss="cross entropy")
        elif self.variant == "kernel":
            explainer = sage.KernelEstimator(imputer, loss="cross entropy")
        else:
            raise ValueError(f"Unsupported SAGE variant: {self.variant}")

        sage_values = explainer(X, y)
        return sage_values.values


def get_explainer(name: str, variant: str):
    """
    Returns the explainer class matching name and variant.

    Must be instantiated with a model before use.

    Example:
        cls = get_explainer("shap", "kernel")
        explainer = cls(model)
    """
    name = name.lower()
    variant = variant.lower()

    if name == "shap":
        return lambda model: SHAPExplainer(model, variant)
    elif name == "sage":
        return lambda model: SAGEExplainer(model, variant)
    else:
        raise ValueError(f"Unsupported explainer: {name}")
