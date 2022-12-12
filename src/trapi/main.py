import logging
import time

from fastapi import Request
from fastapi.responses import RedirectResponse

from drkg_model.api import api as drkg_model_api
from openpredict.config import settings
from openpredict.utils import init_openpredict_dir
from openpredict_evidence_path.api import api as evidence_path_api
from openpredict_explain_shap.api import api as explain_shap_api
from trapi import api_endpoints, trapi
from trapi.openapi_specs import TRAPI

# Other TRAPI project using FastAPI: https://github.com/NCATS-Tangerine/icees-api/blob/master/icees_api/trapi.py

init_openpredict_dir()

log_level = logging.ERROR
if settings.DEV_MODE:
    log_level = logging.INFO
logging.basicConfig(level=log_level)

app = TRAPI()

app.include_router(trapi.app)
app.include_router(api_endpoints.app)
app.include_router(evidence_path_api)
app.include_router(explain_shap_api)
app.include_router(drkg_model_api)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/health", include_in_schema=False)
def health_check():
    """Health check for Translator elastic load balancer"""
    return {'status': 'ok'}


@app.get("/", include_in_schema=False)
def redirect_root_to_docs():
    """Redirect the route / to /docs"""
    return RedirectResponse(url='/docs')
