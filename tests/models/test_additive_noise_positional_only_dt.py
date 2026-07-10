from pyrecest.models.additive_noise import AdditiveNoiseTransitionModel


def test_transition_and_jacobian_accept_positional_only_dt():
    calls = {}

    def transition(state, dt, /):
        calls["transition"] = (state, dt)
        return state + dt

    def jacobian(state, dt, /):
        calls["jacobian"] = (state, dt)
        return dt

    model = AdditiveNoiseTransitionModel(
        transition,
        jacobian=jacobian,
        dt=0.25,
    )

    assert model.evaluate(2.0) == 2.25
    assert model.jacobian(2.0) == 0.25
    assert calls == {
        "transition": (2.0, 0.25),
        "jacobian": (2.0, 0.25),
    }


def test_positional_only_dt_takes_precedence_over_var_keyword_fallback():
    def transition(state, dt, /, **kwargs):
        return state + dt + kwargs["offset"]

    def jacobian(state, dt, /, **kwargs):
        return dt + kwargs["offset"]

    model = AdditiveNoiseTransitionModel(
        transition,
        jacobian=jacobian,
        dt=0.5,
        function_args={"offset": 3.0},
    )

    assert model.evaluate(1.0) == 4.5
    assert model.jacobian(1.0) == 3.5
