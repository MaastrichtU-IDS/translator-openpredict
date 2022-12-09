from reasoner_validator import TRAPIResponseValidator

from openpredict.config import settings

validator = TRAPIResponseValidator(
    trapi_version=settings.TRAPI_VERSION,

    # If omit or set the Biolink Model version parameter to None,
    # then the current Biolink Model Toolkit default release applies
    # biolink_version=settings.BIOLINK_VERSION,

    # 'sources' are set to trigger checking of expected edge knowledge source provenance
    sources={
            # "ara_source": "infores:molepro",
            # "kp_source": "infores:knowledge-collaboratory",
            # "kp_source_type": "primary"
    },
    # Optional flag: if omitted or set to 'None', we let the system decide the
    # default validation strictness by validation context unless we override it here
    strict_validation=None
)
