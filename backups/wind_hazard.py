import numpy as np

def estimate_wind_hazard(translation_speed_kmh, forecast_delta_deg, rmse_deg):
    """
    translation_speed_kmh: last known movement speed
    forecast_delta_deg: magnitude of next predicted step (degrees)
    rmse_deg: model uncertainty in degrees
    
    Returns: (hazard_level, est_wind_kmh, color)
    """

    # 1 deg ~ 111 km
    forecast_delta_km = forecast_delta_deg * 111

    # combine metrics
    combined = (0.6 * translation_speed_kmh +
                0.3 * forecast_delta_km +
                0.1 * (rmse_deg * 111))

    # Hazard thresholds (km/h)
    if combined < 20:
        return ("Low", combined, "green")
    elif combined < 40:
        return ("Moderate", combined, "yellow")
    elif combined < 70:
        return ("High", combined, "orange")
    else:
        return ("Severe", combined, "red")
