This is a Go AI
====
Using two convolutional neural networks and a tree search, this program is able to play go at a considerably advanced level.

To use the program, you first need to generate network connections. There are two options here:

1. Run `python3 policy_network_km.py` and `value_network.py`.
2. Acquire pregenerated connections from [this Dropbox]() (unpack into the same directory as other files).

Then you can play go against the network by running `python3 play.py`.
