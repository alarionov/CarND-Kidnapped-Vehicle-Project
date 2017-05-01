/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;

	std::default_random_engine gen;

	std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_t(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
      Particle particle = {
        -1,
        dist_x(gen),
        dist_y(gen),
        dist_t(gen),
        1.0
      };
      particles.push_back(particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	std::default_random_engine gen;

  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_t(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) < 1e-6) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
      particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
			particles[i].theta += dist_t(gen);
    } else {
			double theta = particles[i].theta + yaw_rate * delta_t;
			particles[i].x += velocity / yaw_rate * (sin(theta) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(theta)) + dist_y(gen);
			particles[i].theta = theta + dist_t(gen);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	for(int i = 0; i < observations.size(); i++) {
		double min_distance = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y);
		observations[i].id = 0;
		for(int j = 1; j < predicted.size(); j++) {
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if(distance < min_distance) {
				min_distance = distance;
				observations[i].id = j;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {
	std::vector<LandmarkObs> predictions;

	for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
		Map::single_landmark_s mark = map_landmarks.landmark_list[i];
		LandmarkObs observation = {-1, mark.x_f, mark.y_f};
		predictions.push_back(observation);
	}

	for (std::vector<Particle>::iterator particle = particles.begin(); particle != particles.end(); ++particle) {
		std::vector<LandmarkObs> new_observations;
		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs observation = observations[j];
			double x = particle->x + observation.x * cos(particle->theta) - observation.y * sin(particle->theta);
			double y = particle->y + observation.x * sin(particle->theta) + observation.y * cos(particle->theta);
			LandmarkObs new_observation = {-1, x, y};
			new_observations.push_back(new_observation);
		}

		dataAssociation(predictions, new_observations);

		double weight = 0;
		for (int j = 0; j < new_observations.size(); j++) {
			LandmarkObs observation = new_observations[j];
			LandmarkObs prediction  = predictions[observation.id];
			double x_diff = (observation.x - prediction.x) * (observation.x - prediction.x) / (std_landmark[0] * std_landmark[0]);
			double y_diff = (observation.y - prediction.y) * (observation.y - prediction.y) / (std_landmark[1] * std_landmark[1]);
  		weight += -0.5 * (log(2 * std_landmark[0] * std_landmark[1] * M_PI) + x_diff + y_diff);
		}
		particle->weight = exp(weight);
	}
}

void ParticleFilter::resample() {
	std::default_random_engine gen;
	double max_weight = 0;
	for (int i = 0; i < num_particles; i++) {
    if (max_weight < particles[i].weight) {
      max_weight = particles[i].weight;
		}
	}

  std::uniform_int_distribution<int>     uniform_start(0, num_particles - 1);
  std::uniform_real_distribution<double> uniform_spin(0, 2 * max_weight);

  std::vector<Particle> new_particles;

  int    index = uniform_start(gen);
  double beta  = 0;

  for (int i = 0; i < num_particles; i++) {
    beta += uniform_spin(gen);
    while (particles[index].weight < beta) {
      beta -= particles[index].weight;
      index = (index + 1) % num_particles;
    }

    Particle particle = { -1, particles[index].x, particles[index].y, particles[index].theta, 1 };
    new_particles.push_back(particle);
  }

  particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
