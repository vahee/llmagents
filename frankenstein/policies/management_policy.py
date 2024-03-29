from typing import Any, Dict, Tuple
from agentopy import IAction, IState, IPolicy, WithActionSpaceMixin, EntityInfo

class ManagementPolicy(WithActionSpaceMixin, IPolicy):

    async def action(self, state: IState) -> Tuple[IAction, Dict[str, Any], Dict[str, Any]]:

        action_name = state.get_item('agent/components/RemoteControl/force_action/name')
        action_args = state.get_item('agent/components/RemoteControl/force_action/args')
        
        state.remove_item('agent/components/RemoteControl/force_action/name')
        state.remove_item('agent/components/RemoteControl/force_action/args')
        
        if not action_name:
            return self.action_space.get_action('nothing'), {}, {}
        
        return self.action_space.get_action(action_name), action_args, {}
    
    def info(self) -> EntityInfo:
        return EntityInfo(
            name="ManagementPolicy",
            version="0.1.0",
            params={}
        )