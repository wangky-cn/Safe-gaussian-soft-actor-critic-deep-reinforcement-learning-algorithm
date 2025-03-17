function Step_Env(x,t)
set_param('dllModel/time','Value',num2str(t));
set_param('dllModel/action','Value',num2str(x));
set_param('dllModel', 'SimulationCommand', 'continue');
end