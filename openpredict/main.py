from fastapi.responses import RedirectResponse
from openpredict.api import explain, models, predict, trapi
from openpredict.loaded_models import PreloadedModels
from openpredict.openapi import TRAPI
from openpredict.utils import init_openpredict_dir

# Other TRAPI project using FastAPI: https://github.com/NCATS-Tangerine/icees-api/blob/master/icees_api/trapi.py

init_openpredict_dir()
PreloadedModels.init()

app = TRAPI(
    baseline_model_treatment='openpredict-baseline-omim-drugbank',
    baseline_model_similarity='drugs_fp_embed.txt',
)

app.include_router(trapi.app)
app.include_router(predict.app)
app.include_router(explain.app)
app.include_router(models.app)


@app.get("/health", include_in_schema=False)
def health_check():
    """Health check for Translator elastic load balancer"""
    return {'status': 'ok'}


@app.get("/", include_in_schema=False)
def redirect_root_to_docs():
    """Redirect the route / to /docs"""
    return RedirectResponse(url='/docs')
