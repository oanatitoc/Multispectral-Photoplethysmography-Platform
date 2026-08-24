# Hardware Documentation

The prototype combines a custom optical PCB with an ESP32-Sparrow Rev. 2 acquisition board.

## Optical PCB

The custom reflective PPG front end contains:

- ams-OSRAM TCS3448 multispectral sensor;
- four white LEDs placed around the sensor;
- current-limiting resistors;
- 1.8 V regulator for the TCS3448 domain;
- I2C level shifting between the 1.8 V sensor domain and the 3.3 V ESP32 domain;
- ESP32 interface header.

The exported hardware documentation includes:

- `MultispectralPPG_schematic.pdf`;
- `MultispectralPPG_PCB_front.pdf`;
- `MultispectralPPG_PCB_bottom.pdf`;
- `Multispectral_PPG_BOM.csv`;
- `MultispectralPPG_3d.png`;
- `enclosure.png`.

These files document the prototype used in the pilot study. If fabrication-ready files are released later, they should be added here as Gerber/drill files for the PCB and STL/STEP files for the enclosure.

## Controller Board

The optical PCB was connected to an ESP32-Sparrow Rev. 2 board. The controller board is an open-hardware design available at:

<https://github.com/dantudose/ESP32-Sparrow-rev2>

## Signal Interface

The optical PCB communicates with the ESP32 over I2C. The main connector exposes:

- `3V3`;
- `GND`;
- `SCL`;
- `SDA`;
- `INT`;
- `GPIO`.

The desktop software receives CSV-formatted samples from the ESP32 over USB serial communication.
