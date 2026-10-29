# RP2350 Hacking Challenge 2 - Here we go again...

Welcome to the *second* Raspberry Pi RP2350 hacking challenge!

Watch our quick explainer video:

[![](assets/video.png)](https://hextree.io/rp2350-hacking-challenge-2)

Our first [Hacking Challenge](https://github.com/raspberrypi/rp2350_hacking_challenge) back in 2024 was both enlightening and enjoyable for us, and garnered positive feedback from the participants and onlookers, so we decided to do it once more 🙃

This second Hacking Challenge focuses on side-channel analysis of the encrypted boot of the RP2350, as used by our newly developed self-decrypting (AES) binary support.

This new support was developed to allow customers to encrypt their application code and data when stored in external flash, and have these loaded into internal SRAM, where our AES software library will decrypt the application in place, using key material and salt squirreled away in the OTP.

We’re looking for challenge participants to use any form of side-channel analysis (Power, EM, timing. Etc) to leak AES key material in a way that will reduce the effective key length to a point where the decryption of the payload is viable.

We have employed various *devious* individuals to add forms of “hardening” to this AES implementation, so we hope it’ll stand up to a high level of analysis and meddling.

We challenge you to prove us wrong! (again)

Unlike the first challenge, we're running this one as a "first across the line" competition, with a single winner (or winning team).  
- It'll run for 3 months (We may choose to extend this if required)  
- The prize will be $20,000 USD  

## Things in scope:
- Any and all forms of side-channel analysis
- Any version of RP2350 or RP2354 (Die version A2/A3 or A4)  
(Feel free to email us at [doh@raspberrypi.com](mailto:doh@raspberrypi.com) with ideas or questions on what could be in scope)
	
## Out of scope:
- Optical / PVC / SEM attacks on the OTP contents or control logic (You know who you are :-P )
- Modifying the provided AES implementation in a way that weakens it
- Using any of the known glitching bugs reported in A2 / A3 silicon to access OTP contents

## How to report findings
If you think you're onto something that looks promising, we'd like to hear from you early.
We'll be looking for you to be able to articulate your findings and any hypotheses you may have, but don't feel like you have to write a dissertation before you get in touch!  
Please email us via [doh@raspberrypi.com](mailto:doh@raspberrypi.com)

## Example code

In this repository you can find a full, already instrumentalized example of our AES implementation:

- The provided aes.S has been modified slightly from the version used by the bootloader [Picotool](https://github.com/raspberrypi/picotool/blob/develop/enc_bootloader/aes.S), to make it easier to set up and run from regular flash
- rp2350_hacking_challenge_2.c contains a small example program that calls into this AES implementation and already provides a trigger for your initial SCA needs
- If you want to trigger somewhere within the assembly, you can simply add "bl trigger" into the assembly

**Note:**: The AES implementation provided here has some fault-injection hardening checks removed and is modified to make side-channel measurements easier. Attacks should also work against the real code when used in the bootloader.

## Keytool

The AES implementation splits the AES key into a 4-way split. To make this easier to work with you can also find `keytool.py` in this repository.

Using `keytool.py` you can split an existing AES key into a 4-way key:

```
$ python3 ./keytool.py encode 0000000000000000000000000000000000000000000000000000000000000000 
66b3ca75e02ad9c8abb06c0b2d297fb660ed5c58c9029ec883f9dbcd2a16195d5e75fadfd32acb297ca03930f1ff08c6714d3f79eb3a26cdc9ef28f553983141
8ae55043c5a225f84ee3b58e01a4c035cae62eba5e622490ce27736a5aa3794053fe7285c513072124ac2f7bb2415adf1c4bf2d87c7b9c6d3643e43356738a86
```

If you need a C-array, you can add the "-c" option:

```
$ python3 ./keytool.py -c encode 0000000000000000000000000000000000000000000000000000000000000000 
\x5a\xad\x34\x6b\x42\x3e\x8d\xff\xe6\xa7\x78\xeb\xfe\x34\xc1\x7f\xb6\x86\x2f\xb5\xd7\xd7\x7f\x11\xdd\x7f\x84\x8c\xbc\x2e\xd4\x28\x04\x20\xfb\x00\x22\xb2\x8d\xb9\xcb\x6f\xd5\x55\xed\xfd\xa3\xec\x14\x85\x20\x2e\x60\x7a\x59\x46\xca\x91\x3e\x77\xbe\x6e\x47\x1f
\xa8\xc2\x2b\x7d\xc6\x6e\xaa\x4d\xc9\x97\x47\xf0\xa7\x3b\xc6\xc0\x4b\x2e\x5b\xe7\xcd\x4b\x8d\x1f\x36\x0a\x4e\x14\xb0\x6f\x98\xec\xef\xce\x3e\x4b\xed\xfe\x66\x60\x93\xb5\xf2\x67\x91\x85\xaa\x4c\xcd\x24\x0c\x46\xe5\xb0\x6d\x4d\x13\x69\x11\xad\x3b\xfd\x70\xa6
```

## Instructions

To build the project, follow these instructions:

```
mkdir build && cd build
PICO_SDK_FETCH_FROM_GIT="on" cmake -DPICO_BOARD=pico2 .. && make
```
One could also include -DPICO_PLATFORM=rp2350 to the cmake call but that is determined automatically.

This will generate a rp2350_hacking_challenge_2_usb.uf2 and rp2350_hacking_challenge_2_uart.uf2, that you can then copy to your RP2350 board.

Usage of the example program is easy, it supports two commands:

- `K` followed by 128 bytes of 4-way key-data (as created by keytool.py) sets the encryption key. The default-key is all 0.
- `E` executes an encryption and then prints out the result as hex. 

## Further reading
Please see [here](aes_report_monospace.md) for a detailed report into the AES hardening work. 

## Acknowledgements
A big thank you to the following folks:
- Mark Owen for his initial implementation.
- Alex Selby for his work hardening the implementation against SCA.
- Thomas Roth and his team at [Hextree](https://www.hextree.io) whom very kindly helped test and develop this second challenge.
- Colin O'Flynn and his team at NewAE for their help and fantastic [ChipWhisperer](https://www.newae.com/chipwhisperer) setup.

It continues to be a pleasure and honor to work with such great people.

## Rules, Terms and Conditions
Please see [here](https://www.raspberrypi.com/rp2350-hacking-challenge-2/) for terms, conditions and rules for this challenge.

