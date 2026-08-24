#include <Arduino.h>
#include <Wire.h>

static constexpr uint8_t TCS3448_ADDR = 0x59;

// Registers
static constexpr uint8_t REG_ENABLE   = 0x80;
static constexpr uint8_t REG_ATIME    = 0x81;
static constexpr uint8_t REG_STATUS2  = 0x90;  // AVALID bit6
static constexpr uint8_t REG_ASTATUS  = 0x94;  // Reading latches all ALS bytes
static constexpr uint8_t REG_CFG1     = 0xC6;  // AGAIN
static constexpr uint8_t REG_LED      = 0xCD;  // LED_ACT + LED_DRIVE
static constexpr uint8_t REG_ASTEP_L  = 0xD4;
static constexpr uint8_t REG_ASTEP_H  = 0xD5;
static constexpr uint8_t REG_CFG20    = 0xD6;  // auto_smux bits[6:5]

// Experimental 18-channel lab defaults.
static constexpr uint8_t ALS_AUTO_SMUX_18CH = 0x60;   // auto_smux = 3
static constexpr uint8_t ALS_ATIME = 9;               // 10 integration steps
static constexpr uint16_t ALS_ASTEP = 359;            // ~10 ms per sub-cycle
static constexpr uint8_t ALS_AGAIN = 0x08;            // moderate gain
static constexpr uint8_t LED_DRIVE = 0x10;            // tune experimentally
static constexpr uint8_t DISCARD_FRAMES_AFTER_LED_TOGGLE = 1;
static constexpr uint32_t EXTRA_SETTLE_MS = 4;

static const char *CHANNEL_NAMES[18] = {
  "FZ", "FY", "FXL", "NIR", "VIS2_C1", "FD_C1",
  "F2", "F3", "F4", "F6", "VIS2_C2", "FD_C2",
  "F1", "F7", "F8", "F5", "VIS2_C3", "FD_C3"
};

struct Frame18 {
  uint8_t astatus = 0;
  uint16_t data[18] = {0};
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

static float integration_time_ms() {
  return (float)(ALS_ATIME + 1) * (float)(ALS_ASTEP + 1) * 2.78e-3f;
}

static uint32_t full_frame_wait_ms() {
  // Datasheet: auto_smux=3 cycles through 3 six-channel conversions before
  // storing all 18 ALS results. We wait a full frame plus a small margin.
  float ms = 3.0f * integration_time_ms() + (float)EXTRA_SETTLE_MS;
  if (ms < 8.0f) ms = 8.0f;
  return (uint32_t)(ms + 0.5f);
}

static bool read18(Frame18 &frame) {
  uint8_t buf[1 + 36] = {0};
  if (!readBlock(REG_ASTATUS, buf, sizeof(buf))) return false;

  frame.astatus = buf[0];
  for (int i = 0; i < 18; i++) {
    uint8_t lo = buf[1 + i * 2];
    uint8_t hi = buf[1 + i * 2 + 1];
    frame.data[i] = (uint16_t)lo | ((uint16_t)hi << 8);
  }
  return true;
}

static bool startAls() {
  return write8(REG_ENABLE, 0x03);  // PON + ALS_EN
}

static bool stopAls() {
  return write8(REG_ENABLE, 0x01);  // PON only
}

static void setLed(bool on) {
  if (on) {
    write8(REG_LED, (uint8_t)(0x80 | (LED_DRIVE & 0x7F)));
  } else {
    write8(REG_LED, 0x00);
  }
}

static bool captureStableFrame18(bool led_on, Frame18 &out) {
  const uint32_t wait_ms = full_frame_wait_ms();
  Frame18 scratch;

  if (!stopAls()) return false;
  setLed(led_on);
  delay(1);
  if (!startAls()) return false;

  for (uint8_t i = 0; i < DISCARD_FRAMES_AFTER_LED_TOGGLE; i++) {
    delay(wait_ms);
    if (!read18(scratch)) return false;
  }

  delay(wait_ms);
  if (!read18(out)) return false;

  return stopAls();
}

static void printHeader() {
  Serial.print("ms,us,off_astatus,on_astatus");
  for (int i = 0; i < 18; i++) {
    Serial.print(',');
    Serial.print(CHANNEL_NAMES[i]);
    Serial.print("_on,");
    Serial.print(CHANNEL_NAMES[i]);
    Serial.print("_off,");
    Serial.print(CHANNEL_NAMES[i]);
    Serial.print("_diff");
  }
  Serial.println();
}

static void printConfigBanner() {
  Serial.print("# mode=18ch_lab auto_smux=3 atime=");
  Serial.print(ALS_ATIME);
  Serial.print(" astep=");
  Serial.print(ALS_ASTEP);
  Serial.print(" again=0x");
  Serial.print(ALS_AGAIN, HEX);
  Serial.print(" led_drive=0x");
  Serial.print(LED_DRIVE, HEX);
  Serial.print(" discard_frames=");
  Serial.print(DISCARD_FRAMES_AFTER_LED_TOGGLE);
  Serial.print(" frame_wait_ms=");
  Serial.println(full_frame_wait_ms());
}

void setup() {
  Serial.begin(115200);
  delay(50);

  Wire.begin(21, 22);
  Wire.setClock(400000);
  delay(10);

  write8(REG_CFG20, ALS_AUTO_SMUX_18CH);
  write8(REG_ATIME, ALS_ATIME);
  write8(REG_ASTEP_L, (uint8_t)(ALS_ASTEP & 0xFF));
  write8(REG_ASTEP_H, (uint8_t)(ALS_ASTEP >> 8));
  write8(REG_CFG1, ALS_AGAIN);
  write8(REG_LED, 0x00);
  write8(REG_ENABLE, 0x01);  // PON

  printConfigBanner();
  printHeader();
}

void loop() {
  Frame18 off_frame;
  Frame18 on_frame;
  if (!captureStableFrame18(false, off_frame)) return;
  if (!captureStableFrame18(true, on_frame)) return;

  uint32_t ms_now = millis();
  uint32_t us_now = micros();

  Serial.print(ms_now);
  Serial.print(',');
  Serial.print(us_now);
  Serial.print(',');
  Serial.print(off_frame.astatus);
  Serial.print(',');
  Serial.print(on_frame.astatus);

  for (int i = 0; i < 18; i++) {
    int32_t diff = (int32_t)on_frame.data[i] - (int32_t)off_frame.data[i];
    Serial.print(',');
    Serial.print(on_frame.data[i]);
    Serial.print(',');
    Serial.print(off_frame.data[i]);
    Serial.print(',');
    Serial.print(diff);
  }
  Serial.println();
}
