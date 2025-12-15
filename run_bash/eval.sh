python cartpole_with_noise.py --mode record --policy ppo --policy-path models/ppo_planb.pt --steps 20000 --plan-b --no-plan-b-reward --force-field --field-strength 1.0 --max-blocks 10 --spawn-prob 0.15 --max-episode-steps 5000 --prefix ppo_planb_eval --horizontal-only


python cartpole_with_noise.py --mode record --policy ppo --policy-path models/ppo.pt --plan-b --no-plan-b-reward --force-field --k-nearest 3 --field-strength 50 --field-pole-strength 50 --max-blocks 5 --spawn-prob 0.15 --max-episode-steps 5000 --horizontal-only --prefix ppo_eval --record-episodes 100 --steps 999999 --acc-min 25 --acc-max 30 --spawn-y-band-half 2 --min-episode-reward-to-save 300 --seed 7 

python cartpole_with_noise.py --mode record --policy ppo 
--policy-path models/ppo.pt --plan-b --no-plan-b-reward 
--force-field --field-strength 50 --field-pole-strength 50 
--max-blocks 5 --spawn-prob 0.15 --max-episode-steps 5000 
--horizontal-only --prefix ppo_eval --record-episodes 10 
--steps 999999 --acc-min 25 --acc-max 30 --spawn-y-band-half 2