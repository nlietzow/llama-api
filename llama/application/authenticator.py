from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.status import HTTP_403_FORBIDDEN

AUTH_TOKEN = "dbd24881e8790182c8e992cc6ad34899"


def verify_token(request: Request):
    if request.headers.get("auth_token") != AUTH_TOKEN:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Not authenticated")
