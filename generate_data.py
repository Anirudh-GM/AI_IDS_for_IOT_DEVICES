import random, csv

with open("sample_iot_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["packet_rate", "avg_packet_size", "label"])
    for _ in range(1000):
        if random.random() < 0.8:
            # Normal behavior
            writer.writerow([random.uniform(1, 10), random.uniform(200, 800), "normal"])
        else:
            # Attack behavior
            writer.writerow([random.uniform(15, 50), random.uniform(800, 2000), "attack"])

print("✅ Dataset created successfully! (sample_iot_data.csv)")
