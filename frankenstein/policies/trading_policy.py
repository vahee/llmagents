from typing import Any, Dict, Tuple
from agentopy import IAction, IState, IPolicy, WithActionSpaceMixin
from frankenstein.lib.trading.schemas import Signal


class TradingPolicy(WithActionSpaceMixin, IPolicy):
    def __init__(self) -> None:
        super().__init__()
        self._lot_size = 0.1
        self._symbol = 'EURUSD'
        self._last_bar_ts = None

        self._open_threshold = 100
        self._close_threshold = 50

        self._sl = 100
        self._tp = 3 * self._sl

    async def action(self, state: IState) -> Tuple[IAction, Dict[str, Any], Dict[str, Any]]:

        signal: Signal = state.get_item('bands/signal')
        
        tp_pips = signal.tp_pips if signal.tp_pips is not None else self._tp
        sl_pips = signal.sl_pips if signal.sl_pips is not None else self._sl

        existing_positions = state.get_item(
            'broker/positions').get(self._symbol, None)

        if existing_positions is not None and existing_positions['is_open']:
            if signal.direction <= -self._close_threshold:
                if existing_positions['is_long']:
                    return self.action_space.get_action('close'), {
                        'symbol': self._symbol,
                        'comment': signal.comment
                    }, {}
            elif signal.direction >= self._close_threshold:
                if not existing_positions['is_long']:
                    return self.action_space.get_action('close'), {
                        'symbol': self._symbol,
                        'comment': signal.comment
                    }, {}
        else:
            if signal.direction >= self._open_threshold:
                if existing_positions is not None and existing_positions['is_long'] and existing_positions['is_open']:
                    return self.action_space.get_action('hold'), {}, {}

                return self.action_space.get_action('open'), {
                    'symbol': self._symbol,
                    'price': state.get_item('broker/ask'),
                    'volume': self._lot_size,
                    'is_long': True,
                    'take_profit_pips': tp_pips,
                    'stop_loss_pips': sl_pips,
                    'comment':signal. comment
                }, {}
            elif signal.direction <= -self._open_threshold:
                if existing_positions is not None and not existing_positions['is_long'] and existing_positions['is_open']:
                    return self.action_space.get_action('hold'), {}, {}

                return self.action_space.get_action('open'), {
                    'symbol': self._symbol,
                    'price': state.get_item('broker/bid'),
                    'volume': self._lot_size,
                    'is_long': False,
                    'take_profit_pips': tp_pips,
                    'stop_loss_pips': sl_pips,
                    'comment': signal.comment
                }, {}
        return self.action_space.get_action('hold'), {}, {}