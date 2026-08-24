#include <Arduino.h>
#include <Wire.h>

static constexpr uint8_t TCS3448_ADDR = 0x59;

// Registers
static constexpr uint8_t REG_ENABLE   = 0x80;
static constexpr uint8_t REG_ATIME    = 0x81;
static constexpr uint8_t REG_STATUS2  = 0x90;  // AVALID - bit6
static constexpr uint8_t REG_ASTEP_L  = 0xD4;
static constexpr uint8_t REG_ASTEP_H  = 0xD5;
static constexpr uint8_t REG_CFG1     = 0xC6;  // AGAIN
static constexpr uint8_t REG_LED      = 0xCD;  // LED_ACT + LED_DRIVE
static constexpr uint8_t REG_CFG20    = 0xD6;  // auto_smux bits[6:5]
static constexpr uint8_t REG_ASTATUS  = 0x94;  // latches dataset when read

static bool write8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(TCS3448_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return (Wire.endTransmission() == 0);
}

static bool read8(uint8_t reg, uint8_t &val) {
  Wire.beginTransmission(TCS3448_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)TCS3448_ADDR, 1) != 1) return false;
  val = (uint8_t)Wire.read();
  return true;
}

static bool readBlock(uint8_t startReg, uint8_t *buf, size_t len) {
  Wire.beginTransmission(TCS3448_ADDR);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0) return false;

  size_t got = Wire.requestFrom((int)TCS3448_ADDR, (int)len);
  if (got != len) return false;

  for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)Wire.read();
  return true;
}

static bool waitAvalid(uint32_t timeout_ms = 80) {
  uint32_t t0 = millis();
  while (millis() - t0 < timeout_ms) {
    uint8_t s2 = 0;
    if (!read8(REG_STATUS2, s2)) return false;
    if (s2 & 0x40) return true;     // bit6 = AVALID
    delay(1);
  }
  return false;
}

// auto_smux=0 => 6 channels: FZ, FY, FXL, NIR, 2xVIS, FD
static bool read6(uint16_t ch[6]) {
  uint8_t buf[1 + 12] = {0}; // ASTATUS + 6*2 bytes
  if (!readBlock(REG_ASTATUS, buf, sizeof(buf))) return false;

  for (int i = 0; i < 6; i++) {
    uint8_t lo = buf[1 + i * 2];
    uint8_t hi = buf[1 + i * 2 + 1];
    ch[i] = (uint16_t)lo | ((uint16_t)hi << 8);
  }
  return true;
}

static bool measure(bool led_on, uint8_t led_drive, uint16_t ch[6]) {
  // stop ALS, keep power on
  if (!write8(REG_ENABLE, 0x01)) return false;

  // set LED
  if (led_on) write8(REG_LED, (uint8_t)(0x80 | (led_drive & 0x7F)));
  else        write8(REG_LED, 0x00);

  // start ALS
  if (!write8(REG_ENABLE, 0x03)) return false;

  // wait measurement complete
  if (!waitAvalid()) return false;

  return read6(ch);
}

void setup() {
  Serial.begin(115200);
  delay(50);

  Wire.begin(21, 22);
  Wire.setClock(400000);
  delay(10);

  // auto_smux = 0 (6-channel) => faster + simpler
  write8(REG_CFG20, 0x00);

  // Integration ~10ms (good compromise)
  // tint = (ATIME+1)*(ASTEP+1)*2.78us
  const uint8_t  atime = 9;     // 10
  const uint16_t astep = 359;   // 180  => ~5ms
  write8(REG_ATIME, atime);
  write8(REG_ASTEP_L, (uint8_t)(astep & 0xFF));
  write8(REG_ASTEP_H, (uint8_t)(astep >> 8));

  // Gain moderate
  write8(REG_CFG1, 0x08); // try 0x08 / 0x09 / 0x0A depending on amplitude

  // Power on (PON)
  write8(REG_ENABLE, 0x01);

  Serial.println(
  "ms,us,"
  "FZ_on,FZ_off,FZ_diff,"
  "FY_on,FY_off,FY_diff,"
  "FXL_on,FXL_off,FXL_diff,"
  "NIR_on,NIR_off,NIR_diff,"
  "VIS2_on,VIS2_off,VIS2_diff,"
  "FD_on,FD_off,FD_diff"
  );
}

void loop() {
  // LED drive: start medium
  const uint8_t led_drive = 0x10;

  uint16_t off[6], on[6];
  if (!measure(false, led_drive, off)) return;
  if (!measure(true,  led_drive, on )) return;

  int32_t diff[6];
  for (int i = 0; i < 6; i++)
    diff[i] = (int32_t)on[i] - (int32_t)off[i];

  uint32_t ms_now = millis();
  uint32_t us_now = micros();

  Serial.print(ms_now); Serial.print(',');
  Serial.print(us_now); Serial.print(',');

  // FZ (0)
  Serial.print(on[0]); Serial.print(','); Serial.print(off[0]); Serial.print(','); Serial.print(diff[0]); Serial.print(',');
  // FY (1)
  Serial.print(on[1]); Serial.print(','); Serial.print(off[1]); Serial.print(','); Serial.print(diff[1]); Serial.print(',');
  // FXL (2)
  Serial.print(on[2]); Serial.print(','); Serial.print(off[2]); Serial.print(','); Serial.print(diff[2]); Serial.print(',');
  // NIR (3)
  Serial.print(on[3]); Serial.print(','); Serial.print(off[3]); Serial.print(','); Serial.print(diff[3]); Serial.print(',');
  // 2xVIS (4)
  Serial.print(on[4]); Serial.print(','); Serial.print(off[4]); Serial.print(','); Serial.print(diff[4]); Serial.print(',');
  // FD (5)
  Serial.print(on[5]); Serial.print(','); Serial.print(off[5]); Serial.print(','); Serial.println(diff[5]);
}
