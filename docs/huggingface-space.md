# Hugging Face Docker Space

CoRe's root `Dockerfile` packages the native C++ backend, FastAPI UI, MuJoCo
models, and a headless OSMesa renderer into one Hugging Face Docker Space.

The public deployment is available at
[huggingface.co/spaces/robotaemoon/CoRe](https://huggingface.co/spaces/robotaemoon/CoRe).

The target selector is populated from CoRe's robot registry and exposes all
eleven bundled models: G1, H1, H2, R1, K1, Apollo, Oli, N1, ADAM Lite, T1, and
PM01. The Docker Space therefore uses the same model assets and retargeting
profiles as the local browser demo and CLI.

## Public demo limits

The image starts one worker on port `7860` with conservative anonymous-service
limits:

- 32 MB maximum SOMA NPZ upload
- 1,800 frames per motion
- three running-plus-queued jobs
- 1280×720 maximum preview resolution
- no intermediate stage archives
- completed uploads and results removed after 30 minutes

The final robot-motion NPZ, manifest, thumbnail, and MP4 remain downloadable
until the job expires. Space restarts also clear the ephemeral job directory.

## Create the Space

1. Create a new Hugging Face Space and select **Docker** as its SDK.
2. Clone the new Space repository.
3. Copy this CoRe checkout into the Space working tree without its `.git`
   directory.
4. Replace the Space's root `README.md` with
   `deploy/huggingface/README.md` so the required Space metadata is retained.
5. Commit and push to the Space repository. Hugging Face builds the root
   `Dockerfile` and serves port `7860` automatically.

The GitHub README intentionally keeps its publication-focused layout; the
Space metadata therefore lives in a separate template.

See Hugging Face's official [Docker Spaces documentation][docker-spaces] and
[first Docker Space guide][first-space] for repository creation and build-log
details. Hugging Face currently requires a PRO, Team, or Enterprise plan to
create a Docker Space; hardware billing and availability can change, so check
the current [Spaces overview][spaces-overview] before deployment.

## Local container check

```bash
docker build --tag core-hf-space .
docker run --rm --publish 7860:7860 core-hf-space
```

Open <http://127.0.0.1:7860> and verify the backend:

```bash
curl http://127.0.0.1:7860/api/health
```

The response should report `"backend": "native"` and the public limits listed
above. The container uses `MUJOCO_GL=osmesa`, so MP4 rendering does not require
a display server or GPU.

## Hardware

Start with CPU Basic for functional testing. CoRe is CPU-bound; CPU Upgrade is
the useful first upgrade when public queue latency becomes too high. A GPU
Space is not required by the current pipeline.

## Persistence and privacy

Do not attach persistent storage for anonymous motion uploads unless a separate
retention and privacy policy is established. The default image deliberately
uses `/tmp/core-runs` and never uploads user motions or results to the Hub.
For a private or persistent service, review Hugging Face's [storage behavior]
[space-storage] and define a retention policy before changing this default.

[docker-spaces]: https://huggingface.co/docs/hub/spaces-sdks-docker
[first-space]: https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo
[spaces-overview]: https://huggingface.co/docs/hub/spaces-overview
[space-storage]: https://huggingface.co/docs/hub/spaces-storage
