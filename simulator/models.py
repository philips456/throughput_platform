from django.db import models


class Antenna(models.Model):
    tower_id = models.IntegerField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    radius = models.FloatField()

    def __str__(self):
        return f"Antenna {self.tower_id}"
