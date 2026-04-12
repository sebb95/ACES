#fejk exempel data
def get_history_data():
    return {
        "trip": {
            "name": "Tur_2026_03_14",
            "start": "02-02-2026",
            "end": "03-03-2026",
            "total_count": 576,
            "total_weight": 1357,
        },
        "sessions": [
            {
                "id": "Økt_001",
                "start": "08:12",
                "end": "08:45",
                "count": 123,
                "weight": 310,
                "corrections": 6,
                "expanded": False,
                "species": [],
            },
            {
                "id": "Økt_002",
                "start": "06:14",
                "end": "09:34",
                "count": 356,
                "weight": 659,
                "corrections": 9,
                "expanded": True,
                "species": [
                    {"name": "Torsk", "count": 129, "weight": 205},
                    {"name": "Sej", "count": 54, "weight": 98},
                    {"name": "Bifangst", "count": 173, "weight": 289},
                ],
            },
        ],
    }