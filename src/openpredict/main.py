from fastapi.responses import RedirectResponse

from drkg_model.api import api as drkg_model_api
from openpredict.api import models, trapi
from openpredict.api.openapi_specs import TRAPI
from openpredict.loaded_models import PreloadedModels
from openpredict.utils import init_openpredict_dir
from openpredict_evidence_path.api import api as evidence_path_api
from openpredict_explain_shap.api import api as explain_shap_api
from openpredict_model.api import api as openpredict_model_api

# Other TRAPI project using FastAPI: https://github.com/NCATS-Tangerine/icees-api/blob/master/icees_api/trapi.py

init_openpredict_dir()
PreloadedModels.init()

app = TRAPI(
    baseline_model_treatment='openpredict-baseline-omim-drugbank',
    baseline_model_similarity='drugs_fp_embed.txt',
)

app.include_router(trapi.app)
app.include_router(openpredict_model_api)
app.include_router(evidence_path_api)
app.include_router(explain_shap_api)
app.include_router(drkg_model_api)
app.include_router(models.app)


@app.get("/health", include_in_schema=False)
def health_check():
    """Health check for Translator elastic load balancer"""
    return {'status': 'ok'}


@app.get("/", include_in_schema=False)
def redirect_root_to_docs():
    """Redirect the route / to /docs"""
    return RedirectResponse(url='/docs')
