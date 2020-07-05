import Environment
import Search
import Threat
import Graph
import numpy as np
import scipy.signal as sg
import random


class Measurement():
    def __init__(self, position, value, noise=0):
        self.position = position
        self.value = value
        self.noise = noise
        self.h = None

    def __str__(self):
        return f"Measurement: {self.position} {self.value}"


class IPAS():
    def __init__(self, environment, sensor_addrs, sensor_noise=0, start_pt=(0,0), end_pt=None, position_converter=None, wait_to_continue=False):
        self.environment = environment
        self.sensor_count = len(sensor_addrs)
        self.sensor_noise = np.diag([sensor_noise for i in range(self.sensor_count)])
        self.wait_to_continue = wait_to_continue
        self.start_pt = start_pt
        self.latest_estimate = [0 for i in range(len(self.environment.threat_field.threats))]
        self.threat_locations = []
        self.threat_shapes = []
        self.true_threats = []
        self.comms = Communications(sensor_addrs, position_converter)
        for i in range(len(self.environment.threat_field.threats)):
            self.threat_locations.append(self.environment.threat_field.threats[i].location)
            self.threat_shapes.append(self.environment.threat_field.threats[i].shape)
            self.true_threats.append(self.environment.threat_field.threats[i].intensity)

    def get_H(self, position):
        h = []
        for threat in self.environment.threat_field.threats:
            h.append(np.exp(-(((position[0]-threat.location[0])**2)+((position[1]-threat.location[1])**2))/(2*threat.shape[0]**2)))
        return h

    def take_measurement(self, position, noise = 0):
        return Measurement(position, self.environment.threat_field.threat_value(position[0], position[1]) + np.random.normal()*noise, noise)

    def update_map(self, measurements):
        h = []
        m = []
        for measurement in measurements:
            h.append(self.get_H(measurement.position))
            m.append(measurement.value)
        h = np.array(h)
        m = np.array(m)
        threat_parameter_mean_estimate = np.linalg.lstsq(h,m, rcond=-1)[0]
        threat_parameter_covar_estimate = h.transpose().dot(self.sensor_noise)
        threat_parameter_covar_estimate = threat_parameter_covar_estimate.dot(h)
        #print(threat_parameter_mean_estimate)
        #print(threat_parameter_covar_estimate)
        a = threat_parameter_covar_estimate.dot(h.transpose())
        b1 = h.dot(threat_parameter_covar_estimate)
        b2 = b1.dot(h.transpose())
        b3 = b2+self.sensor_noise
        #print("\n\n")
        #print(b1)
        #print(b2)
        #print(b3)
        b = (h.dot(threat_parameter_covar_estimate.dot(h.transpose()))+self.sensor_noise)
        #print("\n\n")
        #print(a)
        #print(b)
        k = np.linalg.lstsq(a.transpose(),b,rcond=-1)[0]
        return [threat_parameter_mean_estimate, threat_parameter_covar_estimate, k, h, m]

    def build_with_confidence(self, mean_field_estimate, covar_estimate_matrix, covar_limit = 1):
        #print("mean_field_estimate:", mean_field_estimate)
        covar_estimate = np.diag(covar_estimate_matrix)
        #print("covar_estimate:", covar_estimate)
        if covar_limit!=None:
            for i in range(len(self.latest_estimate)):
                if covar_estimate[i]<covar_limit:
                    self.latest_estimate[i] = mean_field_estimate[0][i]
                else:
                    self.latest_estimate[i] = 0
        else:
            for i in range(len(self.latest_estimate)):
                self.latest_estimate[i] = mean_field_estimate[0][i]

    def path_planing(self):
        threats = []
        latest_estimate_reduced = self.latest_estimate
        for n, threat in enumerate(latest_estimate_reduced):
            if threat < 0:
                latest_estimate_reduced[n] = 0
        for i in range(len(self.latest_estimate)):
            threats.append(Threat.GaussThreat(location = self.threat_locations[i], shape = self.threat_shapes[i], intensity = latest_estimate_reduced[i]))
            #threats.append(Threat.GaussThreat(location = self.threat_locations[i], shape = self.threat_shapes[i], intensity = self.latest_estimate[i]*self.threat_shapes[i]))
        temp_field = Threat.GaussThreatField(threats = threats, offset = 0)

        temp_env = Environment.XYEnvironment(x_size=self.environment.x_size, y_size=self.environment.x_size, n_grid_x=self.environment.n_grid_x, n_grid_y=self.environment.n_grid_y, threat_field=temp_field)

        graph = Graph.Graph(env = temp_env)
        start_node = Graph.XYNode(node_id=0, pos_x=self.start_pt[0], pos_y=self.start_pt[1])
        graph.add_vertex(start_node)
        start_vertex = graph.get_vertex(start_node)
        goal_x, goal_y = temp_env.get_location_from_gridpt(temp_env.n_grid - 1)
        goal_node = Graph.XYNode(temp_env.n_grid - 1, goal_x, goal_y)
        goal_vertex = Graph.Vertex(node=goal_node)
        goal_vertex_found = Search.Astar(graph=graph, start_vertex=start_vertex, goal_vertex=goal_vertex)
        path = [goal_vertex_found.node]
        Search.reconstruct_path(goal_vertex_found, path)
        # for waypoint in path:
        # 	print(waypoint)
        return path

    def reconfigure_sensors(self, path, intensity = 5, reduced_parameters = None):
        if reduced_parameters==None:
            reduced_parameters = self.sensor_count
        proto_map = []
        n_threat_plot = 4 * self.environment.x_size+1
        half_wksp_size = 1
        x_plot, y_plot = np.meshgrid(np.linspace(-half_wksp_size, half_wksp_size, n_threat_plot), np.linspace(-half_wksp_size, half_wksp_size, n_threat_plot))
        for k in range(len(self.threat_locations)):
            proto_map.append(intensity * (1 / (2 * self.threat_shapes[k][0] * self.threat_shapes[k][1]))*np.exp((-1 / 2) *
                (((x_plot - self.threat_locations[k][0]) ** 2) / (self.threat_shapes[k][0] ** 2) +((y_plot - self.threat_locations[k][1]) ** 2) /
                    (self.threat_shapes[k][1] ** 2))))
        max_xcorr_table = np.zeros((len(self.threat_locations), len(path)))
        p_count = 0
        for point in path:
            proto_threat = (intensity * (1 / (2 * self.threat_shapes[k][0] * self.threat_shapes[k][1]))*np.exp((-1 / 2) *
                (((x_plot - point.pos_x) ** 2) / (self.threat_shapes[k][0] ** 2) +
                    ((y_plot - point.pos_y) ** 2) / (self.threat_shapes[k][1] ** 2))))
            for pm in range(len(self.threat_locations)):
                cc = sg.correlate2d(proto_map[pm], proto_threat)
                max_xcorr_table[pm][p_count] = np.amax(np.absolute(cc))
            p_count+=1
        sum_table = np.sum(max_xcorr_table,1)
        indx = np.argsort(sum_table)
        reduced_indx = indx[:reduced_parameters]
        # reduced_indx = indx[-reduced_parameters:]
        print(reduced_indx)
        basis_grid_list = "points close to basis threats ordered by proximity to basis threats"
        new_sensor_positions = []
        for i in range(self.sensor_count):
            new_sensor_positions.append(self.threat_locations[reduced_indx[i]])
        return new_sensor_positions

    def reconfigure_sensors_easy(self, path, intensity=5, reduced_parameters=None):
        if reduced_parameters == None:
            reduced_parameters = self.sensor_count
        distance_min = []
        for threat in self.threat_locations:
            distances = [(threat[0]-node.pos_x)**2+(threat[1]-node.pos_y)**2 for node in path]
            distance_min.append(min(distances))
        indx = np.argsort(distance_min)
        reduced_indx = indx[:reduced_parameters]
        print(reduced_indx)
        basis_grid_list = "points close to basis threats ordered by proximity to basis threats"
        new_sensor_positions = []
        for i in range(self.sensor_count):
            new_sensor_positions.append(self.threat_locations[reduced_indx[i]])
        return new_sensor_positions

    def simulate(self, max_iters=100):
        possible_coordinates = [(x, y) for x in range(2,self.environment.x_size-1) for y in range(2,self.environment.y_size-1)]
        sensor_positions = random.sample(possible_coordinates, self.sensor_count)
        print("initial sensor positions:", sensor_positions)
        first_measurements = []
        for sensor in sensor_positions:
            first_measurements.append(self.take_measurement(sensor, noise = self.sensor_noise[0][0]))
        field_data = self.update_map(first_measurements)
        self.latest_estimate = field_data[0]
        #print("actual threats: "+str(self.true_threats))
        #print("estimated threats: "+str(self.latest_estimate))
        path = self.path_planing()
        new_path = path
        new_sensor_positions = self.reconfigure_sensors_easy(path)
        print("new_sensor_positions:", new_sensor_positions)
        iters = 1
        #while iters<max_iters:
        while iters<max_iters and (new_sensor_positions != sensor_positions or path!=new_path):
            sensor_positions = new_sensor_positions
            new_measurements = []
            for sensor in new_sensor_positions:
                new_measurements.append(self.take_measurement(sensor))
            h = field_data[3]
            new_field_data = self.update_map(new_measurements)
            new_mean_field_estimate = field_data[0] + np.transpose(field_data[2].dot((np.transpose([new_field_data[4]]) - (h.dot(np.transpose([field_data[0]]))))))
            #print("new_mean_field_estimate\n", new_mean_field_estimate)
            try:
                new_covar_estimate = np.linalg.inv(np.linalg.inv(field_data[1]) + h.transpose().dot(self.sensor_noise.dot(h)))
            except:
                new_covar_estimate = np.linalg.pinv(np.linalg.pinv(field_data[1]) + h.transpose().dot(self.sensor_noise.dot(h)))
            self.build_with_confidence(new_mean_field_estimate, new_covar_estimate)
            path = new_path

            new_path = self.path_planing()
            new_sensor_positions = self.reconfigure_sensors_easy(new_path)
            print("new_sensor_positions:", new_sensor_positions)
            field_data = new_field_data
            # print("threat location             threat heaght estimation          threat height actual")
            # for i in range(len(self.true_threats)):
            #     print(str(self.threat_locations[i])+"        "+str(self.latest_estimate[i])+"         "+str(self.true_threats[i]))
            iters+=1
        return path

    def demonstrate(self, max_iters=20):
        possible_coordinates = [(x, y) for x in range(2,self.environment.x_size-1) for y in range(2,self.environment.y_size-1)]
        sensor_positions = random.sample(possible_coordinates, self.sensor_count)
        print("initial sensor positions:", sensor_positions)
        if self.wait_to_continue: input('Press enter to continue...')
        first_measurements = self.comms.send_positions_to_sensors(sensor_positions)
        field_data = self.update_map(first_measurements)
        self.latest_estimate = field_data[0]
        #print("actual threats: "+str(self.true_threats))
        #print("estimated threats: "+str(self.latest_estimate))
        path = self.path_planing()
        new_path = path
        new_sensor_positions = self.reconfigure_sensors_easy(path)
        print("new_sensor_positions:", new_sensor_positions)
        if self.wait_to_continue: input('Press enter to continue...')
        iters = 1
        #while iters<max_iters:
        while new_sensor_positions != sensor_positions and iters<max_iters or path!=new_path:
            sensor_positions = new_sensor_positions
            new_measurements = self.comms.send_positions_to_sensors(new_sensor_positions)
            h = field_data[3]
            new_field_data = self.update_map(new_measurements)
            new_mean_field_estimate = field_data[0] + np.transpose(field_data[2].dot((np.transpose([new_field_data[4]]) - (h.dot(np.transpose([field_data[0]]))))))
            #print("new_mean_field_estimate\n", new_mean_field_estimate)
            try:
                new_covar_estimate = np.linalg.inv(np.linalg.inv(field_data[1]) + h.transpose().dot(self.sensor_noise.dot(h)))
            except:
                pass
            self.build_with_confidence(new_mean_field_estimate, new_covar_estimate)
            path = new_path
            new_path = self.path_planing()
            new_sensor_positions = self.reconfigure_sensors_easy(new_path)
            print("new_sensor_positions:", new_sensor_positions)
            if self.wait_to_continue: input('Press enter to continue...')
            field_data = new_field_data
            print("threat location             threat heaght estimation          threat height actual")
            for i in range(len(self.true_threats)):
                print(str(self.threat_locations[i])+"        "+str(self.latest_estimate[i])+"         "+str(self.true_threats[i]))
            iters+=1
        return path


class Communications:
    def __init__(self, addrs, ipas2pozyx):
        self.sensor_addrs = addrs
        self.ipas2pozyx = ipas2pozyx

    def send_positions_to_sensors(self, positions):
        session = FuturesSession()

        requests = []
        for n, a in enumerate(self.sensor_addrs):
            p = self.ipas2pozyx.convert(positions[n])
            requests.append(session.get(f"http://{a}/image_sensing_value/", timeout=100))

        measurements = []
        for n, req in enumerate(requests):
            j = req.result().json()
            #pos = j['position']
            #threat = j['threat_value']
            threat = j['value']
            measurements.append(Measurement(positions[n], threat))
            print(measurements[-1])
        return measurements
