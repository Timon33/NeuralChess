"""
Serve a live TensorBoard instance using a Modal web endpoint.
"""

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("neuralchess-tensorboard")

    # Define an image with tensorboard and setuptools installed (setuptools provides pkg_resources)
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "tensorboard", "setuptools<70.0.0"
    )

    # Mount the same volume used by training
    volume = modal.Volume.from_name("neuralchess-data", create_if_missing=True)

    class VolumeReloadMiddleware:
        """Middleware to reload the Modal Volume on every HTTP request."""

        def __init__(self, wsgi_app):
            self.wsgi_app = wsgi_app

        def __call__(self, environ, start_response):
            volume.reload()
            return self.wsgi_app(environ, start_response)

    @app.function(
        image=image,
        volumes={"/vol": volume},
        timeout=86400,  # Keep the server alive for up to 24 hours
    )
    @modal.wsgi_app()
    def serve():
        import tensorboard
        from tensorboard.backend.application import TensorBoardWSGIApp

        print("Starting TensorBoard WSGI App...")

        # Ensure we have the latest data before starting
        volume.reload()

        board = tensorboard.program.TensorBoard()
        board.configure(logdir="/vol/checkpoints/tensorboard", load_fast="false")
        (data_provider, deprecated_multiplexer) = board._make_data_provider()

        wsgi_app = TensorBoardWSGIApp(
            board.flags,
            board.plugin_loaders,
            data_provider,
            board.assets_zip_provider,
            deprecated_multiplexer,
            experimental_middlewares=[VolumeReloadMiddleware],
        )

        return wsgi_app
