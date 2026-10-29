```
================================================================================
                          REPORT ON AES HARDENING
                          Alex Selby, January 2025
================================================================================

SUMMARY OF PROJECT
==================

The task was to harden an implementation of AES-256 on a Raspberry Pi RP2350,
to protect it against side-channel vulnerabilities which arise through 
observation of small, high-frequency, changes in the power consumption.

The starting point was an AES implementation due to Mark Owen which included a 
number of options for countermeasures with varying impacts in terms of code 
size and speed. My contribution has been to build a system to detect power 
signal vulnerabilities automatically, which was then used to improve the 
existing countermeasures significantly as well as implement new ones with 
better performance tradeoffs.

NOTATION AND RECAP OF AES-256
=============================

- The text (ciphertext or plaintext) is divided into 16 byte chunks called 
  blocks
- The exclusive OR operation is denoted by ^
- HW refers to Hamming weight, the number of set bits in a binary number, so 
  HW(x) means the number of set bits of the binary number x.
- "Power signal" or "signal" refers to the data-dependent component of the 
  power waveform detected by the ADC. (The power waveform with the average 
  subtracted off, where the average is over all possible relevant values of 
  the registers and memory.)

AES-256 takes a 256-bit key and uses the operation KEYEXPANSION to create 
fifteen 128-bit round keys which are used in the ADDROUNDKEY operation. The 
first two round keys are just the first and second 128-bits respectively of 
the 256-bit key.

For each 16-byte block, AES-256 takes a 128-bit initial state which starts as 
the Initialization Vector "IV", and iterates the following operations on the 
128-bit state:

    ADDROUNDKEY
    SBOX (SUBBYTES)
    SHIFTROWS
    MIXCOLUMNS

for 14 rounds (13 for MIXCOLUMNS, 15 for ADDROUNDKEY), reaching a final 
128-bit state.

In CTR (counter) mode, which is what we're using here, the IV for block b is 
equal to IV0+b, that is, some fixed initial IV, which we call IV0, plus a 
block counter, and the plaintext is derived as 
plaintextblock = ciphertextblock ^ AESfinalstate.

ASSUMPTIONS (GENERALLY HOLDING, THOUGH SEE BELOW)
=================================================

1. There are between 1 and 32768 16-byte blocks to be decrypted (up to 
   512kbytes total)
2. The adversary ("attacker") has access to the source code
3. The attacker has exact timing information and knows what instruction is 
   being executed
4. The attacker can run the decryption operation repeatedly, without limit
5. The attacker can measure power with a device similar to a 
   Chipwhisperer-Husky
6. The attacker knows IV0, the initial IV
7. The attacker knows a partial "crib" (some plaintext and what ciphertext it 
   corresponds to)
8. On boot-up, the RP2350 can produce a random number (64 bits or more) that 
   is not predictable

DISCUSSION REGARDING ASSUMPTIONS
================================

Assumption 3 may not hold if we inject "jitter" into the code. That is, if we 
introduce some code whose timing depends on a random quantity, or otherwise 
make the timing unpredictable, for example by randomly altering the clock 
rate. (Note that the code's timing should not depend on any sensitive 
quantity, such as a key byte, as this would be a significant security hole, 
but it's OK for it to depend on things that are independent of sensitive 
quantities, such as random numbers that aren't used for anything else.) We 
may as well add some jitter to the production code as this is easy to do, has 
relatively low overhead (at least in some versions), and may make the 
attacker's job significantly more awkward. However, the analysis here assumes 
that there is no such jitter. Apart from making the code easier to analyse 
for side-channel leaks, the reason for this is that it is likely that such 
timing variations only provide additive, not multiplicative, security. That 
is, it is not envisaged that a determined attacker would have to try to 
deduce the key from side-channel information while simultaneously battling a 
timing distortion. More likely, they would eventually be able to undo the 
timing distortion by resynchronising to the expected power pattern of 
different instructions (the power consumption of different kinds of 
instructions is very different, but the power consumption of the same 
instruction with different data is only slightly different), and having done 
so, produce a power trace as if it were from an time-undistorted trace, thus 
breaking their task into two independent parts.

Assumption 4 (indefinite decryption) is assumed to hold for the present 
analysis, though perhaps it is worth adding a security feature that gives the 
option of limiting the number of decryptions over the product's lifetime. It 
may happen that for some kinds of end uses it is obvious that you would never 
want more than, say, 10 decryptions in a day with legitimate use. Such a 
limitation would make the device harder to break into with a side-channel 
(power snooping) attack, as such attacks normally require a large number of 
power traces, though of course there may be nothing to stop an attacker 
acquiring more than one device.

Assumption 5 is worth mentioning because, though the Chipwhisperer-Husky (the 
device I have been using) is a good quality ADC capture device, it may be 
that there are better devices that are less noisy or work to a higher 
frequency. Such a device would mean an attacker could probe finer details of 
the power signal and may require fewer traces for a successful attack (though 
I don't know of a feasible attack at present).

Assumption 6: it is traditional in cryptography to assume that the IV0 is not 
secret, and indeed the code does not go to such great lengths to disguise it 
as it does for the 256 bit key. However, it still may be a big obstacle to a 
successful attack if the attacker doesn't know IV0 to start with, and I think 
it is worth treating it as a secret, alongside the key, if this is possible 
to do within the available crypto protocols. Nevertheless, the conservative 
assumption in the present analysis is that the attacker knows the IV, and the 
code is not designed to protect IV0 from power snooping.

EARLY VS LATE ROUND VULNERABILITIES AND GENERAL SECURITY ARGUMENT
=================================================================

A typical round 14 vulnerability works like this. The final AES operation is 
ADDROUNDKEY, so we have penultimatestate ^ roundkey14 = AESfinalstate. If we 
assume (assumption 7) our attacker has a crib, i.e., knows some corresponding 
plaintext and ciphertext from guesswork or otherwise, then 
AESfinalstate = plaintextblock ^ ciphertextblock will be known to them. So 
the attacker knows that unknownpenultimatestate(b) = knownfinalstate(b) ^ 
roundkey14, where the "(b)" indicates that the states depend on the block 
number.

In addition, the power measurement of this operation, if unobscured, will give 
some weak (noisy) information about HW(unknownpenultimatestate(b)). This can 
be averaged over many traces (for each block number, b) to get more accurate 
(less noisy) information about the Hamming weight.

Putting it together, the attacker can eventually learn the value of 
HW(x(b) ^ roundkey14) for each cribbed block, b, where x(b)=penultimatestate(b) 
is a supply of essentially random 128-bit numbers known to the attacker. These 
HWs are enough to uncover roundkey14 after about 130 blocks or so in theory 
(it is sufficient for the set of x(b) to span GF(2)^128). (In fact, because 
the ADDROUNKEY operation is done in 32-bit words, an unobscured such operation 
would likely separately leak information about the HW of each of the four 
32-bit words of the 128-bit word, and we'd only need around 34 blocks, with 
repeated traces of each, to deduce roundkey14.) Assuming something similar can 
be done for roundkey13, it is then easy to invert the KEYEXPANSION operation 
and arrive back at the original 256-bit key. (The KEYEXPANSION operation is 
not itself cryptographically secure, even though it uses SBOX.)

Note that:
- x(b) will vary from block to block even if the plaintext is the same for 
  each block, and will appear completely random, because x(b) is the 
  penultimate state of the AES operation which is seeded by a different IV for 
  each block, and the output of AES is essentially random, and this variation 
  generates different independent pieces of information, and
- the attacker needed some external information, in this case the crib, so 
  that one item in the equation penultimatestate ^ roundkey14 = AESfinalstate, 
  was known. The power trace provided information about the other item, and 
  these combined with the equation told the attacker the final piece of 
  information, roundkey14. It is for this reason that side-channel attacks 
  against AES usually target the first or last rounds, if the inputs or 
  outputs are known.

By contrast, since we're using CTR mode, the corresponding round 0 
relationships have very little variation because the input is virtually 
unchanged. For example the first ADDROUNDKEY operation looks like this:

    IV(b) ^ roundkey0 = nextstate(b)

If the IV is known by the attacker (assumption 6), then again we have an 
equation with one quantity known to the attacker and two unknown. Again we 
assume that after sufficiently many power traces (for each block), the 
Hamming weights of the unknown quantities come to light, and this means that 
the attacker can learn HW(IV(b) ^ roundkey0) for all blocks, b. But because 
IV(b)=IV0+b, and IV is a 128-bit quantity, IV(b) will only vary in the bottom 
15 bits or so (because there are at most 32768 blocks), so in contrast to the 
attack against round 14, all the attacker can learn from the change in 
HW(IV(b) ^ roundkey0) are the bottom 15 bits of the 128-bit roundkey0.

It may also be possible (if the level of the ADC signal corresponding to, say, 
half the bits being set can be determined), from the same power traces, to 
learn the HW of the remaining 17 bits of the bottom 32-bit word of 
IV(b) ^ roundkey0, and the HW of the three 32-bit words constituting the 
higher 96-bits of IV(b) ^ roundkey0, but this is weak information on its own, 
being worth only around 3.5 bits out of 32 (you know how many bits are set, 
but not where they are). Extrapolating to the 256-bit key, that would leave 
it with around 214 bits of entropy, which is still plenty for security. 
(Anything over 100 is plenty, though it's not a bad idea to have some in hand 
in case there are weaknesses you don't know about.)

This means there's an obstacle to attacking round 0 that doesn't arise when 
attacking round 14. The lack of variation of the known quantity means the 
attacker will need to find a lot of different attack points in the code in 
order to build up enough independent information to deduce the keys.

However, there could be a leak point in an early round, e.g. round 3, where 
the variation in the low 15 bits of the IV have been mixed up enough to 
interact with all of the key bits. Then it might be possible to mount a 
Correlation Power Analysis (CPA)-style attack, which would work something 
like this: the attacker would model the power variation at a particular 
target instruction, T, in round 3 to be proportional to, let's say, the 
Hamming weight of the state at that point. Then they could start with a 
random key and try 256 different candidates for a particular byte of that 
key, and for each of these trace through the resulting state over the three 
AES rounds to see which gives the best correlation, over many runs, of Hamming 
weight of the state at instruction T with the measured power at T. Then they 
could pick out the best candidate key byte and repeat for other bytes of the 
key, intending to converge on the full true key.

This relies on "3" being big enough that the effect of changing a particular 
key byte has interacted with the change in the initial state, which is 
confined to the bottom 16 bits or so. But it also relies on "3" not being 
too big, otherwise the AES rounds will mix up the power effect of a single 
key byte (in a trial key which is not completely correct) too chaotically to 
be measurably related to the power effect of that byte in the true key.

It is not obvious whether there is such a suitable happy medium round that 
would enable an attack of this form to be successful, so we sidestep this 
difficult question and argue that, under the recommended countermeasures, the 
security derives from there not being enough information available.

This relies on two sorts of defences. One is that for a given block, we 
arrange it so there are only a few different sorts of things being leaked, 
methods for which are described in more detail in sections below. For 
example, if the power variation is proportional to HW(byte 0 of the state) at 
instruction T0, HW(byte 1 of the state) at instruction T1, and so on up to 
HW(byte 15 of the state) at instruction T15, then that counts as 16 pieces of 
information being leaked. But if instead of that, the power variation at each 
instruction T0, T1, ..., T15 were proportional to 
HW(byte 0)+HW(byte 1)+...+HW(byte 15), then although there are 16 leak 
points, there would only be a single piece of information being leaked, 
namely the total Hamming weight of the whole state.

(Mathematical aside: there are ways to count this number, the effective 
number of pieces of information, based on power observations. One way to do 
this is as follows, amounting to calculating a fuzzy version of the rank of 
the "leak matrix": the code is run many times, and the run instance indexes 
the row, the number of cycles from the start of the code indexes the column, 
and the entries are the observed power. Then one can take the singular value 
decomposition of this matrix and count the number of singular values that are 
too big to be random fluctuations. The advantage of using observed power to 
estimate the number of pieces of information, rather than a theoretical model 
of the ARM's power, is that the true theoretical model is complicated, and we 
may miss something. The disadvantage is that it is only valid up to a 
precision that is governed by the number of runs you are able to make.)

The other defence is to decrypt the blocks in a random order, in such a way 
that the order is not known to an attacker. This means an attacker can't use 
the variation in the initial IV as a means of generating more independent 
sources of information. The attacker only "sees" a superposition of all of 
the blocks.

If there are B blocks and L independent leakage points in each block, then 
there are potentially B*L independent leakage points overall (although in 
practice these points are unlikely all to be independent). Permuting the 
blocks randomly effectively reduces B (which could have been up to 32768) to 
1. It's not possible to reduce L to 0, but the techniques below are intended 
to make L quite small. The intention is that it is small compared with 256, 
the number of bits in the key. It is possible to get more than 1 bit of 
information from a single leakage point, but getting more bits gets 
exponentially harder as you need twice the precision, or four times the as 
many samples, for each new bit, together with an accurate model of how to use 
such new bits.

Of course this is not a watertight proof of security as it relies on certain 
assumptions. Possible vulnerabilities include doing targeted higher order 
(third order or more) attacks, effectively increasing L, or measuring power 
with improved hardware, possibly also increasing L, or somehow correlating 
power at instructions that depend on the block number with other leakage 
points, effectively increasing B.

USE OF SHARES AS A COUNTERMEASURE
=================================

Here shares are briefly described in order to make sense of the next sections. 
Shares and other countermeasures are described more fully in the 
Countermeasures sections below.

One of the principal countermeasures against power snooping used in the 
original code, and in present code, is to use shares (the other main one 
being to randomly permute the order of operations and data). A quantity, X, 
such as the key or 128-bit state, that needs to be hidden is maintained as 
two separate quantities A, B, known as shares of X, that satisfy A^B=X. A and 
B are chosen to be individually uniformly random (even though together they 
are not), so knowledge about only one of them is useless to an attacker. For 
example, A and B could be constructed by letting A be random, and then B=X^A.

It is also possible to use more than one share, e.g., X=A^B^C is a 3-way 
share, where this time any two out of A, B and C are uniformly random and 
provide no information: you'd need to combine all three to provide information 
about the hidden quantity X. The more shares there are, the more security 
there is against power snooping, but obviously the less efficient the 
implementation. Here we use a tradeoff with what you might call 2.5-way 
shares for the state and round keys (explained below), and a 4-way share for 
the (non time-critical) key initialisation.

In a later section we shall describe "accidental unsharing" vulnerabilities, 
where the internal operation of the CPU or memory system effectively 
calculates the exclusive OR of two shares, and emits a power signal based on 
this, despite there being no explicit exclusive OR instruction in the code.

METHODS FOR PROBING VULNERABILITIES
===================================

The tools used to detect leaks evolved into three main kinds: Direct 
correlations, Bitwise correlations and ANOVA, and each of these can be used 
to 1st or 2nd order. In each case, the user specifies in advance the set of 
sensitive quantities to be targeted. For example, a sensitive quantity could 
be K5, the 5th byte of the key, or I10, the 10th byte of the IV, or i5 which 
is shorthand for K5^I5 and is the 5th byte of the input to the first round of 
SBOX, or o7, the 7th byte of the output of the first round of SBOX. Then 
there are combinations of these, such as K5^K7^o5^o7. The AES code under 
analysis is then run many times with random key and IV inputs, and power 
readings are collected.

These three methods can be likened to tuning a radio to listen for a signal 
on a particular frequency (a "frequency" in this analogy could be "K5", the 
5th byte of the key, for example). Direct correlations are the most narrowly 
tuned: they are looking for linear correlations of the power signal (in the 
1st order version) with the Hamming weight of the specified quantity (like 
K5). Bitwise correlations are more general: they will find correlations of 
the power signal with the specified quantity where any bits can be flipped, 
so for example, it will find a linear correlation with the Hamming weight of 
K5^Q where Q is any constant. Finally, ANOVA is the most general and will 
find any relationship at all with the quantity in question (like K5) and the 
power signal. There is a trade-off in that looking for a more general 
relationship means there is less statistical power, so a weaker signal may be 
missed (or more data may need to be collected).

In a little more detail, the direct correlation method is this: let P(i,t) 
denote the measured power at cycle t of the code when executed with random 
input set i, and let A(i,s) denote the Hamming weight of the target quantity 
s on input set i. (For example, s could be "K5^K7" or "I6".) Then for every 
cycle t, we can measure the correlation of P(i,t) with A(i,s) over i and flag 
up the target quantity s and time t, if this correlation is significantly 
higher than you would expect by chance. This is what is meant by "1st order 
correlation". If an instruction handles a target quantity, such as K5, then 
there is a good chance that it will "broadcast" a signal that will be picked 
up by correlating the power with A(i,"K5"). See the section below for an 
example readout. Using shares tends to eliminate such 1st order signals, 
though the section below discusses how this can sometimes go wrong.

Second order correlations look at the power at two different timesteps 
together: we are correlating our target quantity A(i,s) with the correlation 
between P(i,t1) and P(i,t2) for two different timesteps t1 and t2. This 
second-order leakage of information can defeat sharing, which is why we use 
other methods to eliminate, or at least significantly reduce, this source of 
side-channel information.

The idea behind second-order correlations is this: suppose the 5th byte of 
the key, K5, is shared as A^B, and an instruction at time t1 carries out the 
operation X=X^A, and the instruction at t2 carries out Y=Y^B. Suppose that 
because of the nature of the XOR instruction, there is a power fluctuation 
proportional to the number of set bits in A at t1 and the number of set bits 
of B at t2. A and B are individually random, so just looking at time t1 or 
time t2 with a first-order correlation will each be uncorrelated with K5. 
However, if the low bit of K5 is a 0, then the low bits of A and B are either 
both 0 or both 1, so positively correlated, and if the low bit of K5 is a 1, 
then the low bits of A and B will be different, so negatively correlated. 
That means the correlation of P(i,t1) with P(i,t2) will itself be correlated 
with the target quantity A(i,"K5"), so the 2-way sharing has been 
circumvented. And in general, an n-way share could in theory be circumvented 
by using a n-fold correlation.

Note that first order measurements of unsharing vulnerabilities and 
second-order measurements both undo the effect of a share such as K5=A^B, but 
in different ways. If P(X) denotes the power fluctuation due to the bit 
pattern X (which we expect to be roughly proportional to the centred Hamming 
weight of X) then with unsharing, what leaks is P(A^B)=P(K5), whereas in 
second-order analysis, you are correlating P(A) with P(B). Applying random 
rotations to the bits of B (a method used in the code) defeats, or at least 
heavily dilutes, the P(A^B) method, but, on the face of it, does nothing for 
the second order method as it doesn't change P(A) or P(B). However, it is 
possible to have hybrid vulnerabilities where, for example, there are two 
points in the code that each have accidental unshares, and these are in turn 
correlated with each other in a second-order fashion. That means it is always 
worth eliminating lower order vulnerabilities because they may contribute to 
higher order vulnerabilities.

Aside: some authors describe second order attacks using 
(P(A)+P(B)+constant)^2, and in general nth order attacks using 
(P(A1)+...+P(An)+constant)^n, but this is not as effective as using the 
correlation of P(A) and P(B).

However, the higher the order of correlations an attacker uses, the more 
dilute the signal becomes, especially if the operations are using 
words-at-a-time rather than bytes-at-a-time, so as a general principle, word 
operations should be used if possible, and byte operations require more 
protection by other means. (The SBOX, which is necessarily a byte operation, 
is protected by random permutations of the order of bytes being operated on.)

The ANOVA method, which generalises the traditional t-test based method, is 
used to detect arbitrary relationships between the power and target quantity, 
not just a correlation with the Hamming weight. For example, if the power at 
time t tended to spike up when key byte 5 was a multiple of 7 between 100 and 
150, then this method could in principle detect it, as it is looking for all 
possible patterns at once. However, because it is a lot more general, the 
statistical power is reduced: you need more samples to discover the 
relationships. It also requires a lot more computing time to do the analysis 
compared to Hamming weight correlations as above which run nicely optimised 
under the numpy/BLAS framework. Since the large majority of leaks appear to 
show up as Hamming weight correlations, the method used here was to treat 
these as canaries and assume that eliminating these leaks would deal with 
other forms of leaks, while using ANOVA occasionally to check that nothing 
big had been missed.

ACCIDENTAL INTERACTIONS/UNSHARING AND AVOIDING THEM
===================================================

Two shares, A and B, can sometimes undergo an implicit XOR operation. If the 
processor or memory system has some internal state that contains A at one 
point but then changes to contain B, then this broadcasts a small power 
signal proportional to HW(A^B), effectively unsharing A and B and revealing 
something about the hidden value, because it costs energy for a 0 to change 
to a 1 or vice-versa. There are two ways this arises: one is when 
data-processing operations happen close to each other, and the other is when 
there are two load or store operations, with any amount of intervening code.

For example, the original SBOX code included the instructions eor r1,r4,r3 
followed by eor r1,r1,r8, where r4 and r8 are the two shares of one 32-bit 
word of the current state, which is the input to the SBOX. This is an excerpt 
of a readout from 1st order leak detection after 1 million traces:

Cycle Dur        PC   Disassembly                  Signif  Targets
----- ---  --------   ---------------------------  ------  -------
   53   1  20080496   eor.w   r1, r4, r3             -0.1  -
   54   1  2008049a   eor.w   r1, r1, r8              0.4  -
   55   2  2008049e   eor.w   r1, r1, r3, ror #8     11.8  i1,i3,i2,i0
   56      "          "                              20.8  i1,i2,i3
   57   1  200804a2   uxtb    r1, r1                 19.0  i2,i3,i1,i1^i2,i0
   58   2  200804a4   ldrb.w  r1, [ip, r1]           -0.8  -

To explain the above figure: there is one line per cycle, so the fourth line 
is a continuation of the two-cycle long third instruction. The Signif[icance] 
column is in standard deviations and it measures the statistical significance 
of the correlation that it has found. (Anything above 7 or so is highly 
significant.) The targets i0, i1, i2 and i3 are the bytes that make up the 
first state word which is r4^r8 here, and are about to have SBOX applied to 
them. The leak detector has found that at cycles 55, 56 and 57, there is a 
noticeable correlation between the power difference (compared to baseline 
random inputs) and the Hamming weights of i1, i2 and i3 (and i0 and i1^i2 at 
cycles 55 and 57). With enough traces, these leaks would allow you to 
determine something like HW(i1)+HW(i2)+HW(i3), HW(i0) and HW(i1^i2). (That 
is not meant to be apparent from the above readout: it depends on correlation 
coefficients which appear in other readouts not shown here, but we can see 
from the above that it's plausible that the attacker is gaining three 
different pieces of information.)

The antidote in this case is to reorder the instructions to separate r4 and 
r8. (NOP delays between each of the instructions would also work, but would 
obviously be less efficient.) This is the readout from 1 million traces with 
the second and third instructions swapped over:

Cycle Dur        PC   Disassembly                  Signif  Targets
----- ---  --------   ---------------------------  ------  -------
   53   1  20080496   eor.w   r1, r4, r3             -1.4  -
   54   2  2008049a   eor.w   r1, r1, r3, ror #8      0.7  -
   55      "          "                              -1.0  -
   56   1  2008049e   eor.w   r1, r1, r8              1.2  -
   57   1  200804a2   uxtb    r1, r1                 -0.4  -
   58   2  200804a4   ldrb.w  r1, [ip, r1]           -1.6  -

which is now completely quiet (at 1 million traces). In other cases barrier 
instructions are inserted between two accidental unshares to clear registers, 
or to fill them with random numbers.

The following excerpt is an example of unsharing caused by internal state in 
the memory system (for loads). It is from an unprotected version of 
ADDROUNDKEY. The first LDM fetches share A of the round key and the second 
fetches share B. The targets K0, K1, ..., K15 are the bytes of the round key. 
After a two cycle delay from the start of the second LDM, there is a strong 
signal (this is from only 10,000 traces; the correlation is around 0.4 which 
is very high) from the HW of the first word of the round key (represented 
here as HW(K0)+HW(K1)+HW(K2)+HW(K3)), then the next word on the next cycle, 
and so on.

Cycle Dur        PC   Disassembly                     Signif  Targets
----- ---  --------   ------------------------------  ------  -------
   17   4  200805aa   ldmia.w ip!, {r0, r1, r2, r3}     -0.8  -
   18      "          "                                 -0.1  -
   19      "          "                                 -0.2  -
   20      "          "                                  0.8  -
   21   1  200805ae   eors   r4, r0                     -0.2  -
   22   1  200805b0   eors   r5, r1                      1.0  -
   23   1  200805b2   eors   r6, r2                     -0.7  -
   24   1  200805b4   eors   r7, r3                     -0.1  -
   25   4  200805b6   ldmia.w ip!, {r0, r1, r2, r3}      0.6  -
   26      "          "                                  1.0  -
   27      "          "                                 14.3  K3,K2,K0,K1
   28      "          "                                 13.5  K5,K6,K4,K7
   29   1  200805ba   eors.w  r8, r8, r0                15.1  K11,K10,K9,K8
   30   1  200805be   eors.w  r9, r9, r1                14.0  K14,K13,K12,K15
   31   1  200805c2   eors.w  sl, sl, r2                -0.1  -

From this and other evidence, it appears that the striped memory system has 
an internal 32-bit buffer for each of the four possible word offsets modulo 
16 in a given memory bank, when doing a load operation. (Stores work a little 
differently, but there aren't so many critical stores in the code.) In other 
words, it's as if there are four hidden words B0-B3 and the first LDM sets 
B0=roundkeyword0shareA, B1=roundkeyword1shareA, etc.. Then the second LDM 
overwrites B0-B3 with B0=roundkeyword0shareB etc.. Because of the energy cost 
of changing a 0 to a 1 and vice-versa, overwriting B0 has the effect of 
broadcasting the power signal HW(roundkeyword0shareA^roundkeyword0shareB) = 
HW(roundkeyword0), which we see above as the signal K3,K2,K0,K1. It appears 
that there is a separate internal 4-word striped memory buffer for the 
scratch memory and the non-scratch memory, so loads from scratch have to be 
obfuscated by random loads from scratch, and loads from non-scratch 
obfuscated by random loads from non-scratch.

To suppress the signal from two loads directly (not just this one, but 
elsewhere in the code), the best way is to load something random between the 
two loads you want to prevent interacting. Because generating random numbers 
can be slow, the code now has a bank of stored random numbers ("chaff") that 
it often loads from, and refreshes with new random numbers at a rate 
controlled by the configuration parameter REFCHAFF_PERIOD.

The signal can also be suppressed indirectly by randomly permuting the order 
of operations. In this case, the round key words are loaded in a random 
order, which means that the above signal is (a) diluted by junk like 
HW(roundkeyword0shareA^roundkeyword3shareB), and (b) averaged over all words, 
so the attacker only gets the one piece of information HW(128-bit roundkey0) 
rather than the four separate pieces HW(32-bit roundkeyword0), 
HW(32-bit roundkeyword1), etc..

There are other forms of unwanted interaction caused by loads, for example the 
offset register (like R5 in LDR R1,[R2,R5,LSL#2]) also appears to be 
buffered, and can interact with previous uses of an offset register. Similar 
to the load buffer, the best direct antidote appears to be to use a load with 
a random offset between the two loads that you want to prevent interacting.

A consequence of loads generating power signals is that it is necessary to be 
careful about stacking sensitive values. For example, if the (sensitive) 
state is stored in R4-R11, and a subroutine wishes to use R4-R7 for other 
purposes, then it may start with a PUSH {R4-R7} and end with a POP {R4-R7} 
which will fill the memory system's striped buffer with sensitive values. An 
effort has been made as far as possible in this code not to stack sensitive 
values.

COUNTERMEASURES IN PRIOR AND PRESENT MAIN DECRYPTION CODE
=========================================================

The code from July 2024 had several important and useful countermeasures 
against side-channel attacks, most of which have, with adjustments, been 
retained and built upon here. This section is about the main decryption code, 
not the initialisation of round keys and look-up tables.

Summary of prior countermeasures:
- 2-way shares for state and round keys
- ST_VPERM (cyclic permutation of state words)
- ST_HPERM (cyclic permutation of state bytes)
- RK_ROR (cyclic permutation of round key bits within a word)
- CT_BPERM (random permutation of the order of block decryption)
- Optional GEN_RAND_SHA (using SHA-256 to generate random numbers)

Summary of present countermeasures:
- 2-way shares for state and round keys
- Permanent VPERM of state words
- Full 16-way permutation of state bytes
- CT_BPERM with new permutation method
- Barriers to prevent unwanted interactions and accidental unsharing
- Partial third state share, Share C
- Partial third round key share, RK share C
- VPERMs on round key words combined with RK_ROR
- Hybrid gen_rand (SHA-256 or LFSR according to speed/security need)
- Simple ROR#16 on all share Bs

As mentioned above, a principal countermeasure used in the original code, and 
in present code, is to use shares. Instead of working with a 128-bit state, 
X, two 128-bit shares, A and B are used that satisfy A^B=X and where A and B 
are individually uniformly random. The linear AES operations ADDROUNDKEY, 
SHIFTROWS and MIXCOLUMNS can be carried out separately on A and B, but the 
SBOX operation, being non-linear, has to be treated differently, in this case 
using look-up tables whose inputs and outputs are muddled with a random 
share. The round key is also 2-way shared and since this only enters the 
decryption calculation with ADDROUNDKEY which is linear, it means the two 
shares can be treated separately, which is helpful.

If this works as intended, then the hidden quantity (X in the above 
description) would be invisible to 1st order attacks, because A and B are 
individually uniformly random, so there is no point in trying to correlate 
anything of interest (such as the HW of a key byte) with either of them. 
However, it is possible for these shares to become accidentally "unshared" as 
described in the section "Accidental interactions/unsharing and avoiding them".

Another countermeasure called VPERM involves cyclically permuting the state 
word registers. This means that it's not possible to tell which state word a 
given point in the code is working on, which has the effect that the power 
signal from one state word (or a byte within that word) is mixed with the 
signals from all state words (or corresponding bytes within that word). This 
reduces the number of pieces of information that an attacker can obtain and 
so is an effective tool. Cyclic permutations like this commute with SBOX, 
MIXCOLS and SHIFTROWS, meaning that these three operations can all be carried 
out on the permuted state, which is good. However, it doesn't commute with 
ADDROUNDKEY, so the July 2024 code removes the VPERM permutation before 
calling this component of AES. Unfortunately this operation itself creates 
2nd order leakages between VPERM and other parts of the code, and means that 
for part of the time, the state is unobscured by VPERM.

[Side note and confusion avoidance: The term "VPERM" arises from "Vertical 
permutation", which envisages the state words arranged in rows, so 
permutation of words moves them vertically. However, the original official 
description of AES envisages state words in columns.]

The present code adopts a slightly different approach whereby the state is 
permanently VPERMed, and ADDROUNDKEY is modified to work with VPERMed states 
without having to unVPERM the state first. Having paid the price of a 
VPERM-aware ADDROUNDKEY operation (a bit longer and slower), we get almost 
for free that the round keys can be separately VPERMed. Or in other words, if 
the state words are rotated by 'x' (mod 4), and the words of a particular 
share of a round key are rotated by 'y' (mod 4), then we can load the round 
key words permuted by x-y (mod 4), and XOR those into the state words, and 
at no point are the state words or round key words ever unVPERMed into their 
original permutation (except 1/4 of the time by chance). That has the effect 
that state words are all mixed with each other and round key words are mixed 
too, without needing a "risky" operation that unVPERMs them.

Another countermeasure called HPERM involves cyclically permuting bytes 
within state word registers which makes it impossible to tell which byte is 
being operated on at a given point in time in SBOX. However, it is still 
true that if you operate on bytes of a word in sequence then the attacker 
knows that after cyclic permutation, the bytes will still be consecutive 
(wrapping around), so you can get 2nd order signals like 
HW(i0^i1)+HW(i1^i2)+HW(i2^i3)+HW(i3^i0) and HW(i0^i2)+HW(i1^i3) (assuming 
for illustration no VPERM). It is unclear whether these together leak enough 
information to allow a full attack, but the present code uses an enhanced 
version, PERM16, which performs a uniform permutation of all 16 state bytes 
before entering SBOX. To do this conveniently, the (VPERMed) state, which is 
normally held in registers R4-R11, is written out to memory, which requires 
certain protections against power signals due to loads as described in the 
section below.

A new countermeasure in the present code is a partial third share, "Share C". 
This is a random byte which is recorded and XORed into all share A (say) 
state bytes. This gives protection against a large class of side-channel 
probes and comes essentially for free because XORing state with a single byte 
commutes with the operations of ADDROUNDKEY, SHIFTROWS and MIXCOLUMNS, and it 
can be incorporated into the way SBOX is implemented just by XORing Share C 
into three of the bytes that control the look-up tables. If the attacker 
wants to target a single state byte such as i0, then they can no longer rely 
on accidental unsharing (see below) or even a second order leakage - it would 
take at least third order. However, this third share does not give any extra 
protection against target expressions like i0^i2 because the Share C byte is 
the same for all state bytes and would cancel out in i0^i2.

There is also a third partial share for the round keys. This is a 32-bit 
random word that acts as a share that is common to each word of each 
roundkey. Again, the idea is to get some of the benefits of a third share 
with little overhead.

Recall from the section "Types of side-channel vulnerabilities", permuting 
the order in which blocks are decrypted serves to protect against attacks on 
the late rounds of the AES operation. To do this, it is necessary to generate 
a permutation of blocks (up to 32768 of them) that is indistinguishable from 
uniform, fast to generate and does not require any significant extra storage 
space. This kind of permutation is called "oblivious" in the literature (or 
sometimes "format-preserving encryption"), because you want to be able to 
calculate where a given block number is permuted to without having to follow 
the paths of all the other numbers under the permutation (you are oblivious 
to them). The prior code, with the CT_BPERM option, used a power-of-2 length 
linear congruential generator, which is possibly sufficient security though 
it does have some known slight predictable qualities, and the power-of-2 
requirement was somewhat inconvenient. This has been replaced with a 
permutation based on the swap-or-not method, adapted to non-powers-of-2 and 
with the murmur3_32 hash instead of a cryptographically secure hash 
(hopefully giving good practical security in a reasonable time).

The original countermeasure RK_ROR, which used arbitrary rotations of round 
key words is now used in the current code in combination with VPERMs of the 
round key words. That is, the 30 round key shares (groups of 4 words) are 
each given independent cyclic permutations, and the 120 round key words are 
each given independent rotations. This helps protect the round key in the 
ADDROUNDKEY subroutine. This was a cautious choice against a possible threat 
of using 2nd order combinations of round keys across different rounds 
(because the key isn't diffused all that much in AES).

An additional simple measure that costs essentially nothing is keeping all 
share Bs, both state and round keys, rotated by 16. (The round key rotation 
is subsumed in the RK_ROR option, if selected, but it is convenient to match 
the rotation of the state share B.) This reduces the possibility for 
accidental unsharing, because, for example if keyword0shareA accidentally 
interacts with keyword0shareB, then the power broadcast will be 
HW(keyword0shareA^(keyword0shareB ROR#16)) which is meaningless, instead of 
HW(keyword0shareA^keyword0shareB) = HW(keyword0) which is useful information. 
This measure is free because it leaves all operations unchanged except 
SHIFTROWS for which the share B operation has to be altered. But the share A 
and B operations in SHIFTROWS were already written out separately, so 
changing one of them does not affect the code size or execution time.

COUNTERMEASURES IN PRIOR AND PRESENT KEY INITIALISATION CODE
============================================================

The original 256-bit key is split into a 4-way share (128 bytes) in an 
offline process before being stored on the OTP memory, and the key 
initialisation reads this and expands it to fifteen round keys. There is no 
production code that need handle the original 256-bit key. In prior code, the 
round keys were protected by shares and RK_RORs. In present code, they are 
also protected by VPERMs and a third partial share (key share C).

In earlier versions, the initial key reading was a pinch point in terms of 
vulnerability, since most ways to handle the key give something away. This 
was mitigated by reading in words, in a random order, and with careful 
barrier loads, but since the first stage was to read the key and output a 
4-way share, it was decided that it was cleaner and easier to start with a 
4-way share. The information in this is so dilute that any reasonable way of 
handling it is not itself like to give much away, or at least, it's not 
likely to be the most vulnerable part of the code. The earlier versions 
illustrated a general difficulty in guarding against higher order 
correlations: the defensive code can itself be snooped and correlated with 
the critical code.

The key expansion code works with the 4-way shares as intermediate quantities, 
using a particular version of SBOX. The round keys are generated in their 
VPERMed state, which is safer than first generating them unprotected, then 
passing them through VPERM.

BOOTLOADER CODE
===============

This is under discussion, but the current plan is to have the bootloader 
authenticate the AES code and encrypted image before calling the AES 
decryption which converts the encrypted image to code (plaintext), which is 
then "chainloaded". The chainloading process itself also comes with an 
operating system authentication step, so the code/plaintext itself will be 
signed.

The first authentication is necessary because otherwise crafted code may be 
substituted for the AES code, or the bootloader code could be tripped up when 
parsing a crafted ciphertext image. Even allowing the AES decryption code to 
run on a correctly-formatted ciphertext image that is crafted by an attacker 
may compromise security, and even if the resulting plaintext/code is rejected 
by the operating system for execution because it fails a signature. The 
reason is that having a free choice over what is decrypted gives many more 
data-gathering options to an attacker.

CONCLUSION
==========

These measures taken together should make it much harder to extract the 
secret key or the plaintext by using the power side-channel. The fact that it 
is "clean" under the present leak detection methods at 2 million traces to 
1st order, and virtually clean to 2nd order, together with the permutation of 
block decryption order, should be an encouraging sign that it would require 
too much effort to break into the system using power traces, though as ever 
it's not possible to be sure that it isn't vulnerable to a particularly 
determined attack, e.g., a carefully chosen set of third or fourth order 
correlations observed over a long time, or some other artful way to combine 
the disparate small power signals.

It is also worth considering, as mentioned above, the possibility of a faster 
or more sensitive ADC capture device, which may be able to extract finer, 
sub-instruction, details more easily.

The decryption speed with default security settings is about 20,000 cycles 
per 16-byte block, which means about 4.6 seconds to decipher the largest 
possible message of 32,768 blocks at a 144 MHz clock speed.

================================================================================
                                  END OF REPORT
================================================================================
```