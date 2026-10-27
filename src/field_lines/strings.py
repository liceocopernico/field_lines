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
    # presets
    "group_preset": {"it": "Configurazione", "en": "Setup"},
    "preset_empty": {"it": "Vuota", "en": "Empty"},
    "preset_single": {"it": "Carica singola", "en": "Single charge"},
    "preset_dipole": {"it": "Dipolo", "en": "Dipole"},
    "preset_equal": {"it": "Due cariche uguali", "en": "Two equal charges"},
    "preset_quadrupole": {"it": "Quadrupolo", "en": "Quadrupole"},
    "preset_capacitor": {"it": "Condensatore piano", "en": "Parallel plate capacitor"},
    # point charges
    "group_charges": {"it": "Cariche puntiformi", "en": "Point charges"},
    "add_positive": {"it": "+ positiva", "en": "+ positive"},
    "add_negative": {"it": "− negativa", "en": "− negative"},
    "remove": {"it": "Rimuovi", "en": "Remove"},
    "charges_disabled": {
        "it": "Il modello a conduttori contiene solo le armature.",
        "en": "The conductor model contains only the plates.",
    },
    # capacitor
    "group_capacitor": {"it": "Condensatore", "en": "Capacitor"},
    "capacitor_show": {"it": "Mostra il condensatore", "en": "Show the capacitor"},
    "plate_width": {"it": "Larghezza armature", "en": "Plate width"},
    "plate_height": {"it": "Altezza armature", "en": "Plate height"},
    "separation": {"it": "Distanza d", "en": "Separation d"},
    "model": {"it": "Modello", "en": "Model"},
    "model_sigma": {"it": "σ uniforme (sovrapposizione)", "en": "Uniform σ (superposition)"},
    "model_conductor": {"it": "Conduttori a ΔV fissata (Laplace)", "en": "Conductors at fixed ΔV (Laplace)"},
    "sigma": {"it": "Densità σ", "en": "Density σ"},
    "voltage": {"it": "Differenza di potenziale ΔV", "en": "Potential difference ΔV"},
    "plate_sigma": {"it": "Colora le armature con la densità σ", "en": "Colour the plates by their density σ"},
    "readout_E_center": {"it": "E al centro", "en": "E at the centre"},
    "readout_E_ideal": {"it": "σ/ε₀ con σ = Q/A", "en": "σ/ε₀ with σ = Q/A"},
    "readout_dV": {"it": "ΔV tra i centri", "en": "ΔV between the centres"},
    "readout_Q": {"it": "Carica Q", "en": "Charge Q"},
    "readout_C": {"it": "Capacità C", "en": "Capacitance C"},
    "readout_C_ideal": {"it": "ε₀A/d (senza bordi)", "en": "ε₀A/d (no edges)"},
    "computing": {"it": "Calcolo in corso…", "en": "Computing…"},
    "clamped": {
        "it": "d portata a {d} cm: con d più piccola la griglia diventa troppo fine.",
        "en": "d set to {d} cm: a smaller d needs too fine a grid.",
    },
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
        "it": "Clic: linea per il punto · C/M: carica +/− · T: inverti · rotella su una carica: cambia q · R: azzera",
        "en": "Click: line through the point · C/M: +/− charge · T: flip · wheel on a charge: change q · R: reset",
    },
    "hint_3d": {
        "it": "Trascina: ruota · trascina una carica: spostala · clic sul piano colorato: linea per il punto",
        "en": "Drag: rotate · drag a charge: move it · click on the coloured plane: line through the point",
    },
    "readout": {"it": "V = {V}    |E| = {E}", "en": "V = {V}    |E| = {E}"},
    # VTK's font has no "|": E stands for the field's strength
    "bar_E": {"it": "E [V/m]", "en": "E [V/m]"},
    "bar_V": {"it": "V [V]", "en": "V [V]"},
    "bar_slice_E": {"it": "E sul piano [V/m]", "en": "E on the plane [V/m]"},
    "bar_sigma": {"it": "densità di carica [μC/m²]", "en": "charge density [μC/m²]"},  # VTK's font has no σ
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
