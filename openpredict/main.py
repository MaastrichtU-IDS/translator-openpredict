from fastapi.responses import RedirectResponse
from openpredict.api import models, predict, trapi
from openpredict.config import PreloadedModels
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
app.include_router(models.app)



@app.get("/health", include_in_schema=False)
def health_check():
    """Health check for Translator elastic load balancer"""
    return {'status': 'ok'}


@app.get("/", include_in_schema=False)
def redirect_root_to_docs():
    """Redirect the route / to /docs"""
    return RedirectResponse(url='/docs')


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# async def async_reasoner_predict(request_body):
#     """Get predicted associations for a given ReasonerAPI query.

#     :param request_body: The ReasonerStdAPI query in JSON
#     :return: Predictions as a ReasonerStdAPI Message
#     """
#     return post_reasoner_predict(request_body)

# # TODO: get_predict wrapped in ReasonerStdApi
