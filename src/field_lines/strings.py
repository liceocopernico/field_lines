"""Every text the user sees, in Italian and English.

tr(key) returns the text in the current language; set_language switches it.
To change a label, edit it here; the GUI reads every string through tr().
"""

LANGUAGES = {"it": "Italiano", "en": "English"}

STRINGS: dict[str, dict[str, str]] = {
    "window_title": {"it": "Linee di campo elettrico", "en": "Electric field lines"},
    # space
    "group_space": {"it": "Spazio", "en": "Space"},
    "mode_3d": {"it": "3D (spazio)", "en": "3D (space)"},
    "mode_2d": {"it": "2D (piano)", "en": "2D (plane)"},
    # model
    "group_model": {"it": "Modello", "en": "Model"},
    "kind_charges": {"it": "Cariche puntiformi e lamine cariche", "en": "Point charges and charged sheets"},
    "kind_conductors": {"it": "Conduttori a potenziale fissato (Laplace)", "en": "Conductors at fixed potentials (Laplace)"},
    "computing": {"it": "Calcolo in corso…", "en": "Computing…"},
    "updating_views": {"it": "Aggiornamento della vista…", "en": "Updating the view…"},
    "ready": {"it": "Pronto", "en": "Ready"},
    "spheres_touch": {
        "it": "Le sfere non possono toccarsi né toccare le armature: spostale.",
        "en": "Spheres may not touch each other or the plates: move them apart.",
    },
    # presets
    "group_preset": {"it": "Configurazione", "en": "Setup"},
    "preset_empty": {"it": "Vuota", "en": "Empty"},
    "preset_single": {"it": "Carica singola", "en": "Single charge"},
    "preset_dipole": {"it": "Dipolo", "en": "Dipole"},
    "preset_equal": {"it": "Due cariche uguali", "en": "Two equal charges"},
    "preset_quadrupole": {"it": "Quadrupolo", "en": "Quadrupole"},
    "preset_capacitor": {"it": "Condensatore piano", "en": "Parallel plate capacitor"},
    "preset_sphere": {"it": "Sfera metallica", "en": "Metal sphere"},
    "preset_two_spheres": {"it": "Due sfere metalliche", "en": "Two metal spheres"},
    "preset_sphere_in_capacitor": {"it": "Sfera nel condensatore", "en": "Sphere in the capacitor"},
    # point charges
    "group_charges": {"it": "Cariche puntiformi", "en": "Point charges"},
    "add_positive": {"it": "+ positiva", "en": "+ positive"},
    "add_negative": {"it": "− negativa", "en": "− negative"},
    "remove": {"it": "Rimuovi", "en": "Remove"},
    # metallic spheres
    "group_spheres": {"it": "Sfere metalliche", "en": "Metal spheres"},
    "add_sphere": {"it": "+ sfera", "en": "+ sphere"},
    # capacitor
    "group_capacitor": {"it": "Condensatore", "en": "Capacitor"},
    "capacitor_show": {"it": "Mostra il condensatore", "en": "Show the capacitor"},
    "plate_width": {"it": "Larghezza armature", "en": "Plate width"},
    "plate_height": {"it": "Altezza armature", "en": "Plate height"},
    "separation": {"it": "Distanza d", "en": "Separation d"},
    "sigma": {"it": "Densità σ", "en": "Density σ"},
    "voltage": {"it": "Differenza di potenziale ΔV", "en": "Potential difference ΔV"},
    "plate_sigma": {"it": "Colora le armature con la densità σ", "en": "Colour the plates by their density σ"},
    "readout_E_center": {"it": "E al centro", "en": "E at the centre"},
    "readout_E_ideal": {"it": "σ/ε₀ con σ = Q/A", "en": "σ/ε₀ with σ = Q/A"},
    "readout_dV": {"it": "ΔV tra i centri", "en": "ΔV between the centres"},
    "readout_Q": {"it": "Carica Q", "en": "Charge Q"},
    "readout_C": {"it": "Capacità C", "en": "Capacitance C"},
    "readout_C_ideal": {"it": "ε₀A/d (senza bordi)", "en": "ε₀A/d (no edges)"},
    # views
    "group_views": {"it": "Visualizza", "en": "Show"},
    "view_lines": {"it": "Linee di campo", "en": "Field lines"},
    "lines_per_nc": {"it": "linee per nC", "en": "lines per nC"},
    "view_equipotentials": {"it": "Equipotenziali", "en": "Equipotentials"},
    "levels": {"it": "livelli", "en": "levels"},
    "view_arrows": {"it": "Frecce del campo", "en": "Field arrows"},
    "view_slice": {"it": "Piano colorato", "en": "Coloured plane"},
    "slice_V": {"it": "potenziale V", "en": "potential V"},
    "slice_E": {"it": "intensità |E|", "en": "strength |E|"},
    "slice_axis": {"it": "perpendicolare a", "en": "perpendicular to"},
    "slice_position": {"it": "posizione", "en": "position"},
    # animation
    "group_animation": {"it": "Animazione", "en": "Animation"},
    "replay": {"it": "▶ Ripeti la crescita", "en": "▶ Replay the growth"},
    "speed": {"it": "velocità", "en": "speed"},
    # language
    "group_language": {"it": "Lingua", "en": "Language"},
    # status bar and colour bars
    "hint_2d": {
        "it": "Clic: linea per il punto · C/M: carica +/− · S: sfera · T: inverti · rotella su una carica: cambia q · R: azzera",
        "en": "Click: line through the point · C/M: +/− charge · S: sphere · T: flip · wheel on a charge: change q · R: reset",
    },
    "hint_3d": {
        "it": "Trascina: ruota · trascina una carica o una sfera: spostala · clic sul piano colorato: linea per il punto",
        "en": "Drag: rotate · drag a charge or a sphere: move it · click on the coloured plane: line through the point",
    },
    "readout": {"it": "V = {V}    |E| = {E}", "en": "V = {V}    |E| = {E}"},
    # VTK's font has no "|": E stands for the field's strength
    "bar_E": {"it": "E [V/m]", "en": "E [V/m]"},
    "bar_V": {"it": "V [V]", "en": "V [V]"},
    "bar_slice_E": {"it": "E sul piano [V/m]", "en": "E on the plane [V/m]"},
    "bar_sigma": {"it": "densità di carica [μC/m²]", "en": "charge density [μC/m²]"},  # VTK's font has no σ
    "bar_sigma_abs": {"it": "modulo della densità di carica [μC/m²]", "en": "magnitude of the charge density [μC/m²]"},
    # colours of σ
    "sigma_scale": {"it": "Scala di σ", "en": "σ scale"},
    "scale_symmetric": {"it": "simmetrica (segno)", "en": "symmetric (sign)"},
    "scale_contrast": {"it": "contrasto (minimo–massimo)", "en": "contrast (minimum–maximum)"},
    "scale_log": {"it": "logaritmica (modulo)", "en": "logarithmic (magnitude)"},
    "sigma_palette": {"it": "Colori di σ", "en": "σ colours"},
    "palette_RdBu_r": {"it": "rosso-blu", "en": "red-blue"},
    "palette_viridis": {"it": "viridis", "en": "viridis"},
    "palette_plasma": {"it": "plasma", "en": "plasma"},
    "palette_turbo": {"it": "turbo (arcobaleno)", "en": "turbo (rainbow)"},
    "palette_cividis": {"it": "cividis", "en": "cividis"},
    "palette_Greys": {"it": "grigi", "en": "greys"},
}

_language = "it"


def set_language(language: str) -> None:
    global _language
    if language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}, expected one of {list(LANGUAGES)}")
    _language = language


def language() -> str:
    return _language


def tr(key: str, **values) -> str:
    text = STRINGS[key][_language]
    return text.format(**values) if values else text
