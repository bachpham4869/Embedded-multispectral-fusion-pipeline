"""ButtonInput raises RuntimeError when gpiozero is unavailable."""
import pytest
import smartbinocular.controls as controls


def test_button_input_raises_when_gpiozero_absent(monkeypatch):
    monkeypatch.setattr(controls, "_GPIOZERO_OK", False)
    monkeypatch.setattr(controls, "_GpioButton", None)
    with pytest.raises(RuntimeError, match="gpiozero"):
        controls.ButtonInput(pin=17, callback=lambda: None)
