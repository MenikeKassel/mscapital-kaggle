"""Cancel a Kaggle kernel session by numeric session id."""

import argparse

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiCancelKernelSessionRequest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id", type=int)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()
    request = ApiCancelKernelSessionRequest()
    request.kernel_session_id = args.session_id
    with api.build_kaggle_client() as kaggle:
        response = kaggle.kernels.kernels_api_client.cancel_kernel_session(request)

    if response.error_message:
        raise RuntimeError(response.error_message)
    print(f"Cancelled Kaggle kernel session {args.session_id}")


if __name__ == "__main__":
    main()
