import os
from collections import defaultdict

annotations_dir = "..." # путь к директории с аннотациями

all_classes = set()
for filename in os.listdir(annotations_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(annotations_dir, filename), 'r') as file:
            for line in file:
                if line.strip():
                    class_id = int(line.split()[0])
                    all_classes.add(class_id)
print("Все классы в датасете:", sorted(all_classes))

class_counts = defaultdict(int)

for filename in os.listdir(annotations_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(annotations_dir, filename)
        with open(filepath, 'r') as file:
            for line in file:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

print("Количество объектов по классам:")
for class_id, count in sorted(class_counts.items()):
    print(f"Класс {class_id}: {count} объектов")