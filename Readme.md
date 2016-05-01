This is a Go AI
====
Using two convolutional neural networks and a tree search, this program is able to play go at a considerably advanced level.

To use the program, you first need to generate network connections. There are two options here:

1. Run `python3 policy_network_km.py` and `python3 value_network.py`.
2. Acquire pregenerated connections from [this Dropbox](https://www.dropbox.com/s/ehs52ltt2j6tgwn/connections.tar.gz?dl=0) (unpack into the same directory as other files).

Then you can play go against the network by running `python3 play.py`.
