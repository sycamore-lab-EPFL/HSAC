import trimesh
import warp as wp
import warp.sim
import os
import math
import numpy as np

class Example:
    def __init__(self, block_folder='./simulator/cont3Dsimulator/data/arch'):
        builder = wp.sim.ModelBuilder()

        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_bodies = 8
        self.scale = 0.8
        self.ke = 1.0e5
        self.kd = 250.0
        self.kf = 500.0

        # meshes
        objs = []
        for f in os.listdir(block_folder):
            file_path = os.path.join(block_folder, f)
            block = self.load_mesh(file_path)
            b = builder.add_body(
            )

            builder.add_shape_mesh(
                body=b,
                mesh=block,
                pos=wp.vec3(0.0, 0.0, 0.0),
                scale=wp.vec3(self.scale, self.scale, self.scale),
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                density=0,
            )

        # finalize model
        self.model = builder.finalize()
        self.model.ground = True

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.renderer = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def load_mesh(self, file_path):
        trimesh_obj = trimesh.load(file_path, force = 'mesh')
        return wp.sim.Mesh(trimesh_obj.vertices, trimesh_obj.faces.flatten())

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step", active=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=True):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
  


    with wp.ScopedDevice("cuda"):
        wp.init()
        example = Example()

        for _ in range(300):
            example.step()