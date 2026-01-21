import numpy as np
from Perpendicular import Perpendicular

class Drone:
    """Drone class - Exact MATLAB algorithm in Python"""
    
    def __init__(self, index_, position_):
        # Drone properties - EXACT MATLAB fields
        self.index = index_
        self.position = np.array(position_, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.heading = 0.0
        self.path = []  # Will store [position, heading] like MATLAB
        self.vrs = []   # Will store norm(vr) values
        
        # Initialize path like MATLAB: obj.path = [obj.path; [obj.position, obj.heading]];
        self.path.append(np.concatenate([self.position, [self.heading]]))
        self.vrs.append(0.0)
        
        # Control parameters - EXACT MATLAB values
        self.kf = 0.8    # Formation gain
        self.kg = 0.6    # Goal tracking gain
        self.kc = 0.12   # Collision avoidance gain
        self.ko = 1.0    # Obstacle avoidance gain
        
        # Zone parameters - EXACT MATLAB values
        self.ra = 0.3    # Alert radius
        self.rs = 2.0    # Sensing radius
    
    def UpdatePosition(self, vel, dt):
        """EXACT MATLAB UpdatePosition"""
        # MATLAB: obj.position = obj.position + vel*dt;
        self.position = self.position + np.array(vel) * dt
        
        # MATLAB: obj.heading = atan2(vel(2), vel(1));
        self.heading = np.arctan2(vel[1], vel[0])
        
        # MATLAB: obj.velocity = vel;
        self.velocity = np.array(vel)
        
        # MATLAB: obj.path = [obj.path; [obj.position, obj.heading]];
        self.path.append(np.concatenate([self.position, [self.heading]]))
    
    def Behavior(self, drones, model):
        """EXACT MATLAB Behavior function"""
        # MATLAB: l = (model.n+1)/2;
        l = (model['n'] + 1) // 2
        
        if self.index == l:
            # Leader - tracking behavior
            vt, reached = self.Tracking(model)
            vc, vr = self.Collision(drones, model)
            vo = self.Obstacle(model)
            vel = vt + vc + vo
        else:
            # Follower - formation behavior
            vf = self.Formation(drones, model)
            vc, vr = self.Collision(drones, model)
            vo = self.Obstacle(model)
            reached = False
            vel = vf + vc + vo
        
        # MATLAB: obj.vrs = [obj.vrs; norm(vr)];
        self.vrs.append(np.linalg.norm(vr))
        
        # MATLAB: if norm(vel) > 2.0, vel = 2.0*vel/norm(vel); end
        vel_norm = np.linalg.norm(vel)
        if vel_norm > 2.0:
            vel = 2.0 * vel / vel_norm
        
        return vel, reached
    
    def Formation(self, drones, model):
        """EXACT MATLAB Formation behavior"""
        # MATLAB: l = (model.n+1)/2;
        l = (model['n'] + 1) // 2
        
        # Find leader (MATLAB uses 1-based indexing)
        leader_idx = l - 1  # Convert to 0-based indexing
        leader = drones[leader_idx]
        
        # MATLAB: dis = model.d*abs(l-obj.index);
        dis = model['d'] * abs(l - self.index)
        
        # MATLAB: if obj.index < l, ang = drones(l).heading + model.alpha;
        #         else, ang = drones(l).heading - model.alpha; end
        if self.index < l:
            ang = leader.heading + model['alpha']
        else:
            ang = leader.heading - model['alpha']
        
        # MATLAB: pd = drones(l).position + dis*[cos(ang), sin(ang)];
        pd = leader.position + dis * np.array([np.cos(ang), np.sin(ang)])
        
        # MATLAB: vf = -obj.kf*(obj.position - pd) + drones(l).velocity;
        vf = -self.kf * (self.position - pd) + leader.velocity
        
        return vf
    
    def Tracking(self, model):
        """EXACT MATLAB Tracking behavior"""
        epsilon = 0.1  # Goal tolerance
        
        # MATLAB: if norm(model.goal-obj.position) > epsilon
        goal = np.array(model['goal'])
        if np.linalg.norm(goal - self.position) > epsilon:
            # MATLAB: vt = -obj.kg*(obj.position - model.goal);
            vt = -self.kg * (self.position - goal)
            
            # MATLAB: if norm(vt) > 1.0, vt = vt/norm(vt); end
            vt_norm = np.linalg.norm(vt)
            if vt_norm > 1.0:
                vt = vt / vt_norm
            
            reached = False
        else:
            # MATLAB: vt = [0,0]; reached = true;
            vt = np.array([0.0, 0.0])
            reached = True
        
        return vt, reached
    
    def Collision(self, drones, model):
        """EXACT MATLAB Collision avoidance"""
        beta = 2.5  # MATLAB constant
        l = (model['n'] + 1) // 2
        
        vc = np.array([0.0, 0.0])
        vr = np.array([0.0, 0.0])
        
        # MATLAB: for i = 1:model.n
        for i, other_drone in enumerate(drones):
            # MATLAB: if obj.index == i, continue; end
            if self.index == other_drone.index:
                continue
            
            # MATLAB: dis = model.d*abs(i-obj.index);
            dis = model['d'] * abs(other_drone.index - self.index)
            
            # MATLAB: pij = obj.position - drones(i).position;
            pij = self.position - other_drone.position
            norm_pij = np.linalg.norm(pij)
            
            # MATLAB: if (obj.index <= l && i <= l) || (obj.index > l && i >= l)
            if (self.index <= l and other_drone.index <= l) or \
               (self.index > l and other_drone.index >= l):
                # MATLAB (commented out): if norm(pij) < obj.rs
                # MATLAB: vc = vc + obj.kc*abs(norm(pij)-dis)^beta*(1/(norm(pij)-obj.ra)^2)*pij/norm(pij);
                # MATLAB: vr = vr + obj.kc*abs(norm(pij)-dis)^beta*(1/(norm(pij)-obj.ra)^2)*pij/norm(pij);
                if norm_pij > self.ra:  # Avoid division by zero
                    term = (self.kc * abs(norm_pij - dis)**beta * 
                           (1 / (norm_pij - self.ra)**2) * pij / norm_pij)
                    vc += term
                    vr += term
            else:
                # MATLAB: if norm(pij) < obj.rs
                if norm_pij < self.rs and norm_pij > self.ra:
                    # MATLAB: vc = vc + obj.kc*exp(-beta*(norm(pij) - obj.ra))/(norm(pij)-obj.ra)*pij/norm(pij);
                    term = (self.kc * np.exp(-beta * (norm_pij - self.ra)) /
                           (norm_pij - self.ra) * pij / norm_pij)
                    vc += term
        
        return vc, vr
    
    def Obstacle(self, model):
        """EXACT MATLAB Obstacle avoidance"""
        vo = np.array([0.0, 0.0])
        
        # MATLAB: for j = 1:size(model.obstacles,2)
        for j in range(len(model['obstacles'])):
            obstacle = model['obstacles'][j]
            dis = float('inf')
            voj = np.array([0.0, 0.0])
            
            # MATLAB: for i=1:size(obstacle,1)
            for i in range(len(obstacle)):
                # MATLAB: per = Perpendicular(obj.position, obstacle(i,:), obstacle(mod(i,4)+1,:));
                a = obstacle[i]
                b = obstacle[(i + 1) % len(obstacle)]  # mod(i,4)+1 in MATLAB
                per = Perpendicular(self.position, a, b)
                
                # MATLAB: dis_per = norm(obj.position-per);
                dis_per = np.linalg.norm(self.position - per)
                
                # MATLAB: if dis_per < dis, dis = dis_per; n_ = per - obj.position; end
                if dis_per < dis:
                    dis = dis_per
                    n_ = per - self.position
                
                # MATLAB: if dis <= obj.rs, voj = - 1/2*(1/dis^2 - 1/obj.rs^2)*n_/norm(n_); end
                if dis <= self.rs and dis > 0:
                    voj = -0.5 * (1/dis**2 - 1/self.rs**2) * n_ / np.linalg.norm(n_)
            
            # MATLAB: vo = vo + obj.ko*voj;
            vo += self.ko * voj
        
        return vo

if __name__ == "__main__":
    # Test the Drone class
    drone = Drone(1, [0, 0])
    print("Drone class test successful!")