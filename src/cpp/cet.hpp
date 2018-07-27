#ifndef __CET_HPP__
#define __CET_HPP__

#include <unordered_map>

namespace cet {

#include "cet.h"

// TODO: this is broken
std::unordered_map<char const *, uint8_t const *> cmaps;

cmaps["CBD1"] = cet_CBD1;
cmaps["L2"] = L2;
cmaps["CBC1"] = CBC1;
cmaps["L4"] = L4;
cmaps["D1A"] = D1A;
cmaps["L6"] = L6;
cmaps["C1s"] = C1s;
cmaps["CBTC2"] = CBTC2;
cmaps["D1"] = D1;
cmaps["R1"] = R1;
cmaps["CBC2"] = CBC2;
cmaps["D13"] = D13;
cmaps["I2"] = I2;
cmaps["D11"] = D11;
cmaps["L9"] = L9;
cmaps["L18"] = L18;
cmaps["D2"] = D2;
cmaps["L11"] = L11;
cmaps["L1"] = L1;
cmaps["L19"] = L19;
cmaps["C5s"] = C5s;
cmaps["L14"] = L14;
cmaps["L12"] = L12;
cmaps["CBTD1"] = CBTD1;
cmaps["C5"] = C5;
cmaps["L16"] = L16;
cmaps["L3"] = L3;
cmaps["CBL2"] = CBL2;
cmaps["R3"] = R3;
cmaps["L7"] = L7;
cmaps["I3"] = I3;
cmaps["D7"] = D7;
cmaps["CBTC1"] = CBTC1;
cmaps["D9"] = D9;
cmaps["L10"] = L10;
cmaps["D10"] = D10;
cmaps["D3"] = D3;
cmaps["C2s"] = C2s;
cmaps["L8"] = L8;
cmaps["L15"] = L15;
cmaps["R2"] = R2;
cmaps["I1"] = I1;
cmaps["L5"] = L5;
cmaps["C4s"] = C4s;
cmaps["CBTL1"] = CBTL1;
cmaps["L13"] = L13;
cmaps["C4"] = C4;
cmaps["D4"] = D4;
cmaps["C1"] = C1;
cmaps["CBTL2"] = CBTL2;
cmaps["D6"] = D6;
cmaps["D8"] = D8;
cmaps["D12"] = D12;
cmaps["CBL1"] = CBL1;
cmaps["L17"] = L17;
cmaps["C2"] = C2;

}

#endif // __CET_HPP__
