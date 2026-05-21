import spidev
import time
import math

VREF = 3.3
ADC_MAX = 1023

# SCT-013-030: 30A / 1V
SENSOR_A_PER_V = 30.0

spi = spidev.SpiDev()
spi.open(0, 0)          # bus 0, CE0
spi.max_speed_hz = 1350000


def read_adc(channel=0):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    value = ((adc[1] & 3) << 8) | adc[2]
    return value


def adc_to_voltage(value):
    return value * VREF / ADC_MAX


def measure_current(channel=0, samples=1000, delay=0.0005):
    values = []

    for _ in range(samples):
        raw = read_adc(channel)
        voltage = adc_to_voltage(raw)
        values.append(voltage)
        time.sleep(delay)

    # 평균값 = bias 전압, 보통 약 1.65V
    bias = sum(values) / len(values)

    # AC 성분 RMS 계산
    squared_sum = 0.0
    for v in values:
        ac_v = v - bias
        squared_sum += ac_v * ac_v

    vrms = math.sqrt(squared_sum / len(values))

    # SCT-013-030: 1Vrms = 30A
    current = vrms * SENSOR_A_PER_V

    return bias, vrms, current


try:
    while True:
        raw = read_adc(0)
        voltage = adc_to_voltage(raw)

        bias, vrms, current = measure_current(channel=0)

        print(
            f"RAW={raw:4d} | "
            f"V_now={voltage:.3f}V | "
            f"Bias={bias:.3f}V | "
            f"Sensor_Vrms={vrms:.4f}V | "
            f"Current={current:.3f}A"
        )

        time.sleep(0.5)

except KeyboardInterrupt:
    spi.close()
    print("Stopped")
