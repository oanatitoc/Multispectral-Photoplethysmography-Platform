#include <Arduino.h>
#include <Wire.h>

static constexpr uint8_t TCS3448_ADDR = 0x59;

// Registers
static constexpr uint8_t REG_ENABLE   = 0x80;
static constexpr uint8_t REG_ATIME    = 0x81;
static constexpr uint8_t REG_STATUS2  = 0x90;  // AVALID bit6
static constexpr uint8_t REG_ASTATUS  = 0x94;  // Reading latches ALS data
static constexpr uint8_t REG_CFG1     = 0xC6;  // AGAIN
static constexpr uint8_t REG_LED      = 0xCD;  // LED_ACT + LED_DRIVE
static constexpr uint8_t REG_ASTEP_L  = 0xD4;
static constexpr uint8_t REG_ASTEP_H  = 0xD5;
static constexpr uint8_t REG_CFG20    = 0xD6;  // auto_smux bits[6:5]

// 12-channel red/NIR lab: auto_smux=2 gives bank 1 + bank 2.
// This includes NIR from the first bank and F6 (~636 nm) from the second bank.
static constexpr uint8_t ALS_AUTO_SMUX_12CH = 0x40;   // auto_smux = 2

#ifndef PPG_12CH_ATIME
#define PPG_12CH_ATIME 9
#endif

#ifndef PPG_12CH_ASTEP
#define PPG_12CH_ASTEP 359
#endif

#ifndef PPG_12CH_AGAIN
#define PPG_12CH_AGAIN 0x08
#endif

#ifndef PPG_12CH_LED_DRIVE
#define PPG_12CH_LED_DRIVE 0x06
#endif

static constexpr uint8_t ALS_ATIME = PPG_12CH_ATIME;       // 10 integration steps by default
static constexpr uint16_t ALS_ASTEP = PPG_12CH_ASTEP;      // ~10 ms per 6-channel sub-cycle
static constexpr uint8_t ALS_AGAIN = PPG_12CH_AGAIN;       // tune experimentally
static constexpr uint8_t LED_DRIVE = PPG_12CH_LED_DRIVE;   // dimmer default avoids F6 saturation
static constexpr uint32_t FRAME_MARGIN_MS = 4;
static constexpr uint8_t STARTUP_DISCARD_FRAMES = 3;

static const char *CHANNEL_NAMES[12] = {
  "FZ", "FY", "FXL", "NIR", "VIS2_C1", "FD_C1",
  "F2", "F3", "F4", "F6", "VIS2_C2", "FD_C2"
};

struct Frame12 {
  uint8_t astatus = 0;
  uint16_t data[12] = {0};
};

static bool write8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(TCS3448_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
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

  for (size_t i = 0; i < len; i++) {
    buf[i] = (uint8_t)Wire.read();
  }
  return true;
}

static bool waitAvalid(uint32_t timeout_ms = 120) {
  uint32_t t0 = millis();
  while (millis() - t0 < timeout_ms) {
    uint8_t status2 = 0;
    if (!read8(REG_STATUS2, status2)) return false;
    if (status2 & 0x40) return true;
    delay(1);
  }
  return false;
}

static float integration_time_ms() {
  return (float)(ALS_ATIME + 1) * (float)(ALS_ASTEP + 1) * 2.78e-3f;
}

static uint32_t full_frame_wait_ms() {
  // auto_smux=2 completes two 6-channel sub-cycles before all 12 values are current.
  float ms = 2.0f * integration_time_ms() + (float)FRAME_MARGIN_MS;
  if (ms < 8.0f) ms = 8.0f;
  return (uint32_t)(ms + 0.5f);
}

static void setLedOn() {
  write8(REG_LED, (uint8_t)(0x80 | (LED_DRIVE & 0x7F)));
}

static bool read12(Frame12 &frame) {
  uint8_t buf[1 + 24] = {0}; // ASTATUS + 12*2 bytes
  if (!readBlock(REG_ASTATUS, buf, sizeof(buf))) return false;

  frame.astatus = buf[0];
  for (int i = 0; i < 12; i++) {
    uint8_t lo = buf[1 + i * 2];
    uint8_t hi = buf[1 + i * 2 + 1];
    frame.data[i] = (uint16_t)lo | ((uint16_t)hi << 8);
  }
  return true;
}

static void printHeader() {
  Serial.print("ms,us,astatus");
  for (int i = 0; i < 12; i++) {
    Serial.print(',');
    Serial.print(CHANNEL_NAMES[i]);
  }
  Serial.println();
}

static void printConfigBanner() {
  Serial.print("# mode=red_nir_12ch_always_on auto_smux=2 atime=");
  Serial.print(ALS_ATIME);
  Serial.print(" astep=");
  Serial.print(ALS_ASTEP);
  Serial.print(" again=0x");
  Serial.print(ALS_AGAIN, HEX);
  Serial.print(" led_drive=0x");
  Serial.print(LED_DRIVE, HEX);
  Serial.print(" frame_wait_ms=");
  Serial.println(full_frame_wait_ms());
}

void setup() {
  Serial.begin(115200);
  delay(50);

  Wire.begin(21, 22);
  Wire.setClock(400000);
  delay(10);

  write8(REG_ENABLE, 0x01);                 // PON only while configuring
  write8(REG_CFG20, ALS_AUTO_SMUX_12CH);
  write8(REG_ATIME, ALS_ATIME);
  write8(REG_ASTEP_L, (uint8_t)(ALS_ASTEP & 0xFF));
  write8(REG_ASTEP_H, (uint8_t)(ALS_ASTEP >> 8));
  write8(REG_CFG1, ALS_AGAIN);
  setLedOn();
  write8(REG_ENABLE, 0x03);                 // PON + ALS_EN

  Frame12 scratch;
  for (uint8_t i = 0; i < STARTUP_DISCARD_FRAMES; i++) {
    delay(full_frame_wait_ms());
    read12(scratch);
  }

  printConfigBanner();
  printHeader();
}

void loop() {
  Frame12 frame;

  delay(full_frame_wait_ms());
  if (!waitAvalid()) return;
  if (!read12(frame)) return;

  uint32_t ms_now = millis();
  uint32_t us_now = micros();

  Serial.print(ms_now);
  Serial.print(',');
  Serial.print(us_now);
  Serial.print(',');
  Serial.print(frame.astatus);

  for (int i = 0; i < 12; i++) {
    Serial.print(',');
    Serial.print(frame.data[i]);
  }
  Serial.println();
}
