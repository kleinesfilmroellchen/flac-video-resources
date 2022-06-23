# SPDX-License-Identifier: Apache-2.0
# This file is partially used as a exponential golomb demonstrator (run it directly as a Python script).

from math import *
from typing import Sequence

'''
Structure of an Exponential Golomb order k encoded number:
----------------------------------------------------------
  000...000   qqq...qqq   rrrrrr...rrrrrr
 \___   ___/ \___   ___/ \______   ______/
     \ /         \ /            \ /
      '           '              '
  # of bits   "quotient"     remainder
   -1 in q    power of k  (always k bits)

The quotient is encoded with 1 added so that 0 can be represented as (encoded)
1. This is the same as Exponential Golomb order 0 encoding, or Elias γ (Gamma)
coding if we didn't add 1.
'''


def encode_exp_golomb_0(number: int) -> str:
    ''' Order 0 Exponential Golomb of x is identical to the Elias gamma code of x+1.'''
    to_encode = number + 1
    # The number of bits that are used when to_encode is written in binary.
    leading_bits = int(floor(log2(to_encode)) + 1)
    # In a real encoder, we would write out leading_bits zeroes followed by to_encode in big endian bit order.
    # This string visualizes what the bits would look like when we read them back in.
    return '{:1} {:b}'.format('0'*(leading_bits-1), to_encode)


def encode_exp_golomb(number: int, k: int) -> str:
    '''
    The fast way (?) to encode a number in exponential Golomb order k:
    Encode number + 2^k in Elias gamma coding.
    '''
    to_encode = number + (2 ** k)
    # number of leading zeroes; need to subtract k off the final result
    N = int(floor(log2(to_encode)))
    # up to this point we've only been doing Elias gamma coding
    return '{:1} {:b}'.format('0'*(N-k),  to_encode)


def alternate_encode_exp_golomb(number: int, k: int) -> str:
    '''
    This way is more understandable:
    Split the number in two, above k and below k bits.
    Encode the upper bits (arbitrarily many!) with Exponential Golomb order 0, as we saw before.
    Include the lower k bits (always exactly k!) unencoded.
    '''
    return ('{:1} {:0' + str(k) + 'b}').format(
        encode_exp_golomb_0(int(floor(number / (2**k)))),
        number % (2**k))


def create_exp_golomb_table(numbers_to_encode: Sequence[int], ks: Sequence[int], fstr: str = '{:>21}') -> Sequence[Sequence[str]]:
    rows = []
    for number in numbers_to_encode:
        rows.append([str(number),
                     fstr.format(encode_exp_golomb_0(number)),
                     *(fstr.format(alternate_encode_exp_golomb(number, k)) for k in ks)])
    return rows


if __name__ == '__main__':
    fstr = '{:>21}'
    # Play with these two variables
    numbers_to_encode = list(range(0, 16, 1))
    numbers_to_encode.extend(range(16, 512, 8))
    ks = range(1, 7)

    print(' num', fstr.format('0'), *map(lambda k: fstr.format(str(k)), ks))
    print('-'*180)
    for number in numbers_to_encode:
        print('{:4}'.format(number), fstr.format(encode_exp_golomb_0(number)), *
              (fstr.format(alternate_encode_exp_golomb(number, k)) for k in ks))
