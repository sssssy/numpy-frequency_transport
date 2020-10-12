#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "bmpFile.cpp"
#include "fft.cpp"

using namespace std;

class Spectrum
{
    public:
    Spectrum(int reso)
    {
        this->reso = reso;
        data = (unsigned char *)calloc(
            sizeof(unsigned char), (int)(pow(reso, 4)));
        Fdata = (unsigned char *)calloc(
            sizeof(unsigned char), (int)(pow(reso, 4)));
    }
    ~Spectrum(){}

    int reso;
    // BYTE *data = (unsigned char*) malloc(sizeof(unsigned char)
    //                  * int(pow(reso, 4)));
    BYTE *data; // x * theta * y * phi
    BYTE *Fdata;
    char* lastop = "NULL";

    void setZero()
    {
        BYTE *pData;
        for (pData = data; pData < data + (int)pow(reso, 4); pData++){
            *(pData) = (BYTE)255;
        }
        lastop = "setZero";
        printf("setZero.\n");
    }

    void setRect()
    {
        for (int i = 0; i < reso; i++)
            for (int j = 0; j < reso; j++)
                for (int k = 0; k < reso; k++)
                    for (int l = 0; l < reso; l++){
                        if(i>0.4*reso && i<0.6*reso 
                            && k>0.4*reso && k<0.6*reso)
                            data[((i*reso+j)*reso+k)*reso+l] = (BYTE)255;
                        else
                            data[((i * reso + j) * reso + k) * reso + l] = (BYTE)0;
                    }

        lastop = "setRect";
        printf("setRect.\n");
    }

    void fft()
    {
        memcpy(Fdata, data, sizeof(Fdata));
        Fdata = fft(Fdata, )
    }

    void save()
    {
        char *fileName = (char*)malloc(sizeof(char)*20);
        fileName = strcpy(fileName, "images/");
        fileName = strcat(fileName, lastop);
        fileName = strcat(fileName, ".bmp");
        BYTE *saveImg = (unsigned char *)malloc(
            sizeof(unsigned char)*(int)(pow(reso, 2)));
        BYTE *pSave = saveImg;

        for(int i=0; i<reso; i++)
            for (int j=0; j < reso; j++)
                *(pSave++) = data[((i * reso) * reso + j) * reso];
        Rmw_Write8BitImg2BmpFile(saveImg, reso, reso, 
            fileName);
        printf("saved.\n");

        delete pSave;
        delete fileName;
    }
};

