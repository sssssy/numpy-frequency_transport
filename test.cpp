#include "Spectrum.h"

int main()
{
    Spectrum s = Spectrum(100);
    s.setRect();
    s.save();
    s.fft();
    s.save();

    delete &s;

    return 0;
}