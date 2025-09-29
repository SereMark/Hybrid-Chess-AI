from __future__ import annotations

import contextlib
import csv
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class MetricsReporter:
    csv_path: str

    def append(self, row: Mapping[str, object], field_order: Sequence[str] | None = None) -> None:
        if not self.csv_path:
            return

        try:
            directory = os.path.dirname(self.csv_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            fieldnames = list(field_order) if field_order is not None else list(row.keys())

            write_header = not os.path.isfile(self.csv_path)

            if not write_header:
                try:
                    with open(self.csv_path, newline="", encoding="utf-8") as existing:
                        reader = csv.reader(existing)
                        header = next(reader, [])
                    if list(header) != fieldnames:
                        backup_path = self.csv_path + ".bak"
                        with contextlib.suppress(Exception):
                            os.replace(self.csv_path, backup_path)
                        write_header = True
                except Exception:
                    write_header = True

            with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            print(f"[MetricsReporter] Failed to append row: {exc}")
