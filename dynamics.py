import numpy as np
import kinpy as kp
import os

class Dynamics:
    def __init__(self, urdf_path):
        # 1. URDF 파일 로드 (Kinpy 방식)
        with open(urdf_path, 'rb') as file: # 'rb'는 바이트 모드 (b 추가)
            urdf_data = file.read()
        
        self.chain = kp.build_serial_chain_from_urdf(
            data=urdf_data, 
            root_link_name="BASE", 
            end_link_name="END_EFFECTOR" 
        )
        
        # 2. 파라미터 설정
        self.dt = 0.1
        self.damping = 1e-4  # DLS 감쇠 계수
        
        # 3. 관절 위치 한계 (Kinpy는 자동으로 파싱해주지 않으므로 수동 설정 필요)
        # Gen3 Lite는 대부분 무한회전이거나 -150~150도 정도입니다.
        # 안전을 위해 -2pi ~ 2pi로 대략 설정하거나, 필요시 값을 직접 수정하세요.
        n_dof = len(self.chain.get_joint_parameter_names())
        self.q_min = -2 * np.pi * np.ones(n_dof)
        self.q_max =  2 * np.pi * np.ones(n_dof)
        
        # 충돌 모델은 Kinpy에서 지원하지 않으므로 생략

    def remove_adjacent_links_from_collision(self):
        """
        Kinpy는 충돌 감지 기능이 없습니다.
        코드 호환성을 위해 함수 껍데기만 남겨둡니다.
        """
        print("[Warn] Kinpy는 충돌 감지를 지원하지 않아 해당 기능이 비활성화됩니다.")
        pass

    def check_self_collision(self, q):
        """
        Kinpy에서는 충돌 체크 불가 -> 무조건 False(충돌 없음) 반환
        """
        return False

    def solve_ik(self, q, u_task):
        """
        현재 각도 q와 목표 작업공간 속도 u_task(6x1)를 받아
        관절 속도 dq를 반환 (Damped Least Squares)
        """
        # 1. 자코비안 계산 (Kinpy)
        # kinpy는 q를 리스트나 numpy array로 받음
        J = self.chain.jacobian(q) # (6 x N) 행렬
        
        # 2. Damped Least Squares (DLS) IK 풀이
        # 수식: dq = J.T * (J*J.T + lambda^2*I)^-1 * u_task
        JJT = np.dot(J, J.T)
        damp_matrix = (self.damping ** 2) * np.eye(6)
        
        # np.linalg.solve는 Ax = B를 풀어줍니다. (inv를 직접 쓰는 것보다 빠르고 정확)
        # (JJT + damp) * temp = u_task
        temp = np.linalg.solve(JJT + damp_matrix, u_task)
        dq = np.dot(J.T, temp)

        # 3. [Hardware Safety] 관절 속도 물리적 한계 클리핑
        #-----------안전장치-----------------------------------------
        joint_vel_limit = 0.2  # rad/s (사용자가 설정한 값)
        
        # 1) 단순 클리핑 (방향이 바뀔 수 있음)
        # dq = np.clip(dq, -joint_vel_limit, joint_vel_limit)
        
        # 2) 비율 스케일링 (방향 유지, 추천)
        max_vel = np.max(np.abs(dq))
        if max_vel > joint_vel_limit:
            scale = joint_vel_limit / max_vel
            dq = dq * scale
        #-----------------------------------------------------------
        
        return dq

    def step(self, q_curr, u_task):
        """제어 입력을 받아 다음 관절 상태를 계산합니다."""
        # 1. IK를 통해 관절 속도 계산
        dq = self.solve_ik(q_curr, u_task)
        
        # 2. 적분 (Integration): q_next = q + dq * dt
        q_next = q_curr + dq * self.dt

        #-----------안전장치 (위치 제한)-------------------------------
        q_next = np.clip(q_next, self.q_min, self.q_max)
        #-----------------------------------------------------------
        
        # 3. 정기구학 (Forward Kinematics) - 다음 위치 확인용
        trans = self.chain.forward_kinematics(q_next)
        
        # 4. 데이터 포맷 변환
        # Kinpy는 Rotation을 Quaternion(w,x,y,z)으로 줍니다. -> Matrix(3x3)로 변환 필요
        pos = trans.pos
        rot_matrix = self._quat_to_rot_matrix(trans.rot)
        
        return q_next, pos, rot_matrix, dq

    def _quat_to_rot_matrix(self, q):
        """
        내부 헬퍼 함수: 쿼터니언 [w, x, y, z] -> 3x3 회전 행렬 변환
        """
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
