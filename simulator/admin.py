from django.contrib import admin
from .models import Antenna


@admin.register(Antenna)
class AntennaAdmin(admin.ModelAdmin):
    list_display = ("tower_id", "longitude", "latitude", "radius")
    search_fields = ("tower_id",)
