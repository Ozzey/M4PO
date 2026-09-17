# Container integration

The simulator-free mock implementation needs only the packages in
`requirements.txt`. IsaacLab containers are release- and driver-specific; add
this repository to the matching NVIDIA IsaacLab image, install it editable,
and configure an external `isaaclab_factory` as described in
`docs/DEVELOPMENT.md`.
