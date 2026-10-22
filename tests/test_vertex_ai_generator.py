"""Test generator"""

import unittest
from copy import deepcopy
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, patch

import kfp
import yaml
from kedro.pipeline import Pipeline, node

from kedro_vertexai.config import PluginConfig
from kedro_vertexai.constants import (
    KEDRO_CONFIG_RUN_ID,
    KEDRO_GLOBALS_PATTERN,
    KEDRO_VERTEXAI_RUNNER_CONFIG,
)
from kedro_vertexai.generator import PipelineGenerator
from kedro_vertexai.vertex_ai.runner import VertexAIPipelinesRunner
from tests.utils import environment


def identity(input1: str):
    return input1  # pragma: no cover


class TestGenerator(unittest.TestCase):
    def create_pipeline(self):
        return Pipeline(
            [
                node(identity, "A", "B", name="node1", tags=["foo", "group.nodegroup"]),
                node(identity, "B", "C", name="node2", tags=["bar", "group.nodegroup"]),
            ]
        )

    def test_should_group_when_enabled(self):
        # given
        expected1 = {
            "cpuLimit": 0.1,
            "cpuRequest": 0.1,
            "resourceCpuLimit": "100m",
            "resourceCpuRequest": "100m",
        }
        expected2 = {
            "cpuLimit": 0.4,
            "cpuRequest": 0.4,
            "memoryLimit": 68.719476736,
            "memoryRequest": 68.719476736,
            "resourceCpuLimit": "400m",
            "resourceCpuRequest": "400m",
            "resourceMemoryLimit": "64Gi",
            "resourceMemoryRequest": "64Gi",
        }
        base = {
            "grouping": {"cls": "kedro_vertexai.grouping.TagNodeGrouper"},
            "resources": {"__default__": {"cpu": "100m"}},
        }
        tags = ["node1", "nodegroup", "foo", "bar", "group.nodegroup"]

        configs = []
        for tag in tags:
            testcfg = deepcopy(base)
            testcfg["resources"][tag] = {"cpu": "400m", "memory": "64Gi"}
            configs.append(testcfg)

        expected = [expected1] + 4 * [expected2]
        for cfg, exp in zip(configs, expected):
            with self.subTest(
                msg=str(next(key for key in cfg["resources"] if key != "__default__"))
            ):
                self.create_generator(config=cfg)
                # when
                with patch(
                    "kedro.framework.project.pipelines",
                    new=self.pipelines_under_test,
                ):
                    pipeline = self.generator_under_test.generate_pipeline(
                        "pipeline", "unittest-image", "MLFLOW_TRACKING_TOKEN"
                    )
                    with NamedTemporaryFile(
                        mode="rt", prefix="pipeline", suffix=".yaml"
                    ) as spec_output:
                        kfp.compiler.Compiler().compile(pipeline, spec_output.name)
                        with open(spec_output.name) as f:
                            pipeline_spec = yaml.safe_load(f)

                        component_args = pipeline_spec["deploymentSpec"]["executors"][
                            "exec-nodegroup"
                        ]["container"]["args"][0]
                        assert (
                            '--nodes "node1,node2"' in component_args
                            or '"node2,node1"' in component_args
                        )

                        self.assertDictEqual(
                            pipeline_spec["deploymentSpec"]["executors"][
                                "exec-nodegroup"
                            ]["container"]["resources"],
                            exp,
                        )

    def test_should_not_add_resources_spec_if_not_requested(self):
        # given
        self.create_generator(
            config={
                "resources": {
                    "__default__": {"cpu": None, "memory": None},
                }
            }
        )

        # when
        with patch(
            "kedro.framework.project.pipelines",
            new=self.pipelines_under_test,
        ):
            pipeline = self.generator_under_test.generate_pipeline(
                "pipeline", "unittest-image", "MLFLOW_TRACKING_TOKEN"
            )
            with NamedTemporaryFile(
                mode="rt", prefix="pipeline", suffix=".yaml"
            ) as spec_output:
                kfp.compiler.Compiler().compile(pipeline, spec_output.name)
                with open(spec_output.name) as f:
                    pipeline_spec = yaml.safe_load(f)

            # then
            for component in ["exec-node1", "exec-node2"]:
                spec = pipeline_spec["deploymentSpec"]["executors"][component][
                    "container"
                ]
                assert "resources" not in spec

    def test_should_add_resources_spec(self):
        # given
        self.create_generator(
            config={
                "resources": {
                    "__default__": {"cpu": "100m"},
                    "node1": {"cpu": "400m", "gpu": "1", "memory": "64Gi"},
                },
                "node_selectors": {
                    "node1": {"cloud.google.com/gke-accelerator": "NVIDIA_TESLA_K80"},
                },
            }
        )

        # when
        with patch(
            "kedro.framework.project.pipelines",
            new=self.pipelines_under_test,
        ):
            pipeline = self.generator_under_test.generate_pipeline(
                "pipeline", "unittest-image", "MLFLOW_TRACKING_TOKEN"
            )
            with NamedTemporaryFile(
                mode="rt", prefix="pipeline", suffix=".yaml"
            ) as spec_output:
                kfp.compiler.Compiler().compile(pipeline, spec_output.name)
                with open(spec_output.name) as f:
                    pipeline_spec = yaml.safe_load(f)

                    # then
                    component1_resources = pipeline_spec["deploymentSpec"]["executors"][
                        "exec-node1"
                    ]["container"]["resources"]
                    assert component1_resources["cpuLimit"] == 0.4
                    assert component1_resources["memoryLimit"] == 68.719476736
                    assert component1_resources["cpuRequest"] == 0.4
                    assert component1_resources["memoryRequest"] == 68.719476736
                    assert component1_resources["resourceCpuLimit"] == "400m"
                    assert component1_resources["resourceCpuRequest"] == "400m"
                    assert component1_resources["resourceMemoryLimit"] == "64Gi"
                    assert component1_resources["resourceMemoryRequest"] == "64Gi"
                    assert component1_resources["accelerator"]["count"] == "1"
                    assert (
                        component1_resources["accelerator"]["type"]
                        == "NVIDIA_TESLA_K80"
                    )

                    component2_resources = pipeline_spec["deploymentSpec"]["executors"][
                        "exec-node2"
                    ]["container"]["resources"]
                    assert component2_resources["cpuLimit"] == 0.1
                    assert component2_resources["cpuRequest"] == 0.1
                    assert component2_resources["resourceCpuLimit"] == "100m"
                    assert component2_resources["resourceCpuRequest"] == "100m"

    def test_should_set_description(self):
        # given
        self.create_generator(config={"description": "DESC"})

        # when
        with patch(
            "kedro.framework.project.pipelines",
            new=self.pipelines_under_test,
        ):
            pipeline = self.generator_under_test.generate_pipeline(
                "pipeline", "unittest-image", "MLFLOW_TRACKING_TOKEN"
            )

            # then
            assert pipeline.description == "DESC"

    @patch("kedro_vertexai.generator.is_mlflow_enabled", return_value=True)
    def test_should_add_env_and_pipeline_in_the_invocations(
        self, mock_is_mlflow_enabled
    ):
        # given
        self.create_generator()

        # when
        with patch(
            "kedro.framework.project.pipelines",
            new=self.pipelines_under_test,
        ):
            pipeline = self.generator_under_test.generate_pipeline(
                "pipeline", "unittest-image", "MLFLOW_TRACKING_TOKEN"
            )
            with NamedTemporaryFile(
                mode="rt", prefix="pipeline", suffix=".yaml"
            ) as spec_output:
                kfp.compiler.Compiler().compile(pipeline, spec_output.name)
                with open(spec_output.name) as f:
                    pipeline_spec = yaml.safe_load(f)

                assert (
                    "kedro vertexai -e unittests mlflow-start"
                    in pipeline_spec["deploymentSpec"]["executors"][
                        "exec-mlflow-start-run"
                    ]["container"]["args"][0]
                )

    def test_should_add_runner_and_runner_config(self):
        # given
        self.create_generator()

        # when
        with patch(
            "kedro.framework.project.pipelines",
            new=self.pipelines_under_test,
        ):
            pipeline = self.generator_under_test.generate_pipeline(
                "pipeline", "unittest-image", "MLFLOW_TRACKING_TOKEN"
            )
            with NamedTemporaryFile(
                mode="rt", prefix="pipeline", suffix=".yaml"
            ) as spec_output:
                kfp.compiler.Compiler().compile(pipeline, spec_output.name)
                with open(spec_output.name) as f:
                    pipeline_spec = yaml.safe_load(f)

            # then
            assert all(
                check
                in pipeline_spec["deploymentSpec"]["executors"]["exec-node1"][
                    "container"
                ]["args"][0]
                for check in (
                    f"{KEDRO_CONFIG_RUN_ID}=",
                    f"{KEDRO_VERTEXAI_RUNNER_CONFIG}='{{",
                    f"--runner {VertexAIPipelinesRunner.runner_name()}",
                )
            )

    def test_should_not_dump_params_even_if_params_are_set(self):
        # Ki fork (PC-3990): should_add_params is forced to False, so params are never
        # dumped via initialize-job nor passed as --config.
        self.create_generator(params={"my_params1": 1.0, "my_param2": ["a", "b", "c"]})

        with patch(
            "kedro.framework.project.pipelines",
            new=self.pipelines_under_test,
        ):
            pipeline = self.generator_under_test.generate_pipeline(
                "pipeline", "unittest-image", "MLFLOW_TRACKING_TOKEN"
            )
            with NamedTemporaryFile(
                mode="rt", prefix="pipeline", suffix=".yaml"
            ) as spec_output:
                kfp.compiler.Compiler().compile(pipeline, spec_output.name)
                with open(spec_output.name) as f:
                    pipeline_spec = yaml.safe_load(f)

            args = pipeline_spec["deploymentSpec"]["executors"]["exec-node1"][
                "container"
            ]["args"][0]
            assert "initialize-job" not in args
            assert 'kedro run -e unittests --pipeline pipeline --nodes "node1"' in args
            assert "--config config.yaml" not in args

    def test_should_add_globals_env_if_present(self):
        with environment({"KEDRO_GLOBALS_PATTERN": "*globals.yml"}):
            self.create_generator(
                params={"my_params1": 1.0, "my_param2": ["a", "b", "c"]}
            )
            self.mock_mlflow(False)
            with patch(
                "kedro.framework.project.pipelines",
                new=self.pipelines_under_test,
            ):

                pipeline = self.generator_under_test.generate_pipeline(
                    "pipeline",
                    "unittest-image",
                    "MLFLOW_TRACKING_TOKEN",
                )
                with NamedTemporaryFile(
                    mode="rt", prefix="pipeline", suffix=".yaml"
                ) as spec_output:
                    kfp.compiler.Compiler().compile(pipeline, spec_output.name)
                    with open(spec_output.name) as f:
                        pipeline_spec = yaml.safe_load(f)

                expected = f'{KEDRO_GLOBALS_PATTERN}="*globals.yml"'
                assert pipeline_spec["deploymentSpec"]["executors"]["exec-node1"][
                    "container"
                ]["args"][0]

                assert (
                    pipeline_spec["deploymentSpec"]["executors"]["exec-node1"][
                        "container"
                    ]["args"][0].count(expected)
                    == 1
                ), "Globals variable should be added once, for kedro run (no initialize-job, PC-3990)"

    @patch("kedro_vertexai.generator.is_mlflow_enabled", return_value=True)
    def test_should_add_host_aliases_if_requested(self, mock_is_mlflow_enabled):
        # given
        self.create_generator(
            config={
                "network": {
                    "host_aliases": [
                        {
                            "ip": "10.10.10.10",
                            "hostnames": ["mlflow.internal", "mlflow.cloud"],
                        }
                    ]
                }
            }
        )

        # when
        with patch(
            "kedro.framework.project.pipelines",
            new=self.pipelines_under_test,
        ):
            pipeline = self.generator_under_test.generate_pipeline(
                "pipeline", "unittest-image", "MLFLOW_TRACKING_TOKEN"
            )
            with NamedTemporaryFile(
                mode="rt", prefix="pipeline", suffix=".yaml"
            ) as spec_output:
                kfp.compiler.Compiler().compile(pipeline, spec_output.name)
                with open(spec_output.name) as f:
                    pipeline_spec = yaml.safe_load(f)

            # then
            hosts_entry_cmd = (
                "echo 10.10.10.10\tmlflow.internal mlflow.cloud >> /etc/hosts;"
            )
            assert (
                hosts_entry_cmd
                in pipeline_spec["deploymentSpec"]["executors"][
                    "exec-mlflow-start-run"
                ]["container"]["args"][0]
            )
            assert (
                hosts_entry_cmd
                in pipeline_spec["deploymentSpec"]["executors"]["exec-node1"][
                    "container"
                ]["args"][0]
            )

    def _compile_pipeline_spec(self):
        with patch(
            "kedro.framework.project.pipelines",
            new=self.pipelines_under_test,
        ):
            pipeline = self.generator_under_test.generate_pipeline(
                "pipeline", "unittest-image", "MLFLOW_TRACKING_TOKEN"
            )
            with NamedTemporaryFile(
                mode="rt", prefix="pipeline", suffix=".yaml"
            ) as spec_output:
                kfp.compiler.Compiler().compile(pipeline, spec_output.name)
                with open(spec_output.name) as f:
                    return yaml.safe_load(f)

    @patch("kedro_vertexai.generator.is_mlflow_enabled", return_value=True)
    def test_should_run_machine_typed_node_as_custom_job(self, _):
        # given
        self.create_generator(
            config={
                "service_account": "pipelines@test-project.iam.gserviceaccount.com",
                "network": {"vpc": "projects/123/global/networks/my-vpc"},
                "resources": {"__default__": {"cpu": "100m"}},
                "machine_types": {"node1": {"machine_type": "n4-highmem-8"}},
            }
        )
        unwrapped_spec = None
        with patch.object(self.generator_under_test.run_config, "machine_types", {}):
            unwrapped_spec = self._compile_pipeline_spec()

        # when
        pipeline_spec = self._compile_pipeline_spec()

        # then
        executors = pipeline_spec["deploymentSpec"]["executors"]
        assert (
            "google-cloud-pipeline-components"
            in executors["exec-node1"]["container"]["image"]
        )
        params = pipeline_spec["components"]["comp-node1"]["inputDefinitions"][
            "parameters"
        ]
        (worker_pool,) = params["worker_pool_specs"]["defaultValue"]
        assert worker_pool["machine_spec"] == {"machine_type": "n4-highmem-8"}
        assert worker_pool["disk_spec"] == {
            "boot_disk_type": "hyperdisk-balanced",
            "boot_disk_size_gb": 100,
        }
        unwrapped = unwrapped_spec["deploymentSpec"]["executors"]["exec-node1"][
            "container"
        ]
        assert worker_pool["container_spec"]["image_uri"] == "unittest-image"
        assert worker_pool["container_spec"]["command"] == unwrapped["command"]
        assert worker_pool["container_spec"]["args"] == unwrapped["args"]
        assert (
            params["service_account"]["defaultValue"]
            == "pipelines@test-project.iam.gserviceaccount.com"
        )
        assert (
            params["network"]["defaultValue"] == "projects/123/global/networks/my-vpc"
        )
        node1_inputs = pipeline_spec["root"]["dag"]["tasks"]["node1"]["inputs"][
            "parameters"
        ]
        assert node1_inputs["mlflow_run_id"]["taskOutputParameter"] == {
            "outputParameterKey": "mlflow_run_id",
            "producerTask": "mlflow-start-run",
        }
        assert executors["exec-node2"]["container"]["image"] == "unittest-image"
        assert executors["exec-node2"]["container"]["resources"]["cpuLimit"] == 0.1

    def test_should_fail_when_params_clash_with_custom_job_inputs(self):
        self.create_generator(
            config={
                "service_account": "pipelines@test-project.iam.gserviceaccount.com",
                "machine_types": {"node1": {"machine_type": "n4-highmem-8"}},
            }
        )
        with patch(
            "kedro.framework.project.pipelines",
            new=self.pipelines_under_test,
        ), self.assertRaisesRegex(ValueError, r"\['location'\].*node1"):
            self.generator_under_test.generate_pipeline(
                "pipeline",
                "unittest-image",
                "MLFLOW_TRACKING_TOKEN",
                params="location:str,my_param:int",
            )

    def test_should_fail_when_machine_type_conflicts_with_resources(self):
        self.create_generator(
            config={
                "service_account": "pipelines@test-project.iam.gserviceaccount.com",
                "resources": {"__default__": {"cpu": "100m"}, "node1": {"cpu": "4"}},
                "machine_types": {"node1": {"machine_type": "n4-highmem-8"}},
            }
        )
        with self.assertRaisesRegex(ValueError, "node1"):
            self._compile_pipeline_spec()

    def test_generate_params_signature(self):
        self.create_generator()

        params = ""
        assert self.generator_under_test._generate_params_signature(params) == ""

        params = "test_param:str"
        assert (
            self.generator_under_test._generate_params_signature(params)
            == "test_param: str"
        )

        params = "test_param:str,param2:int"
        assert (
            self.generator_under_test._generate_params_signature(params)
            == "test_param: str, param2: int"
        )

    def test_add_mlflow_param_to_signature(self):
        self.create_generator()

        params_signature = ""
        assert (
            self.generator_under_test._add_mlflow_param_to_signature(params_signature)
            == "mlflow_run_id: Union[str, None] = None"
        )

        params_signature = "test_param: str"
        assert (
            self.generator_under_test._add_mlflow_param_to_signature(params_signature)
            == "test_param: str, mlflow_run_id: Union[str, None] = None"
        )

        params_signature = "test_param: str, param2: int"
        assert (
            self.generator_under_test._add_mlflow_param_to_signature(params_signature)
            == "test_param: str, param2: int, mlflow_run_id: Union[str, None] = None"
        )

    def test_generate_pipeline_with_params(self):
        self.create_generator()
        with patch(
            "kedro.framework.project.pipelines",
            new=self.pipelines_under_test,
        ):

            pipeline = self.generator_under_test.generate_pipeline(
                pipeline="pipeline",
                image="unittest-image",
                token="MLFLOW_TRACKING_TOKEN",
                params="test_param:str,param2:int",
            )
            with NamedTemporaryFile(
                mode="rt", prefix="pipeline", suffix=".yaml"
            ) as spec_output:
                kfp.compiler.Compiler().compile(pipeline, spec_output.name)
                with open(spec_output.name) as f:
                    pipeline_spec = yaml.safe_load(f)

                    pipeline_params = pipeline_spec["root"]["inputDefinitions"][
                        "parameters"
                    ]
                    assert pipeline_params["test_param"]["parameterType"] == "STRING"
                    assert (
                        pipeline_params["param2"]["parameterType"] == "NUMBER_INTEGER"
                    )

                    assert (
                        "test_param"
                        in pipeline_spec["root"]["dag"]["tasks"]["node1"]["inputs"][
                            "parameters"
                        ]
                    )
                    assert (
                        "param2"
                        in pipeline_spec["root"]["dag"]["tasks"]["node1"]["inputs"][
                            "parameters"
                        ]
                    )

                    assert (
                        "test_param"
                        in pipeline_spec["root"]["dag"]["tasks"]["node2"]["inputs"][
                            "parameters"
                        ]
                    )
                    assert (
                        "param2"
                        in pipeline_spec["root"]["dag"]["tasks"]["node2"]["inputs"][
                            "parameters"
                        ]
                    )

                    assert (
                        "--params test_param={{$.inputs.parameters['test_param']}},param2={{$.inputs.parameters['param2']}}"  # noqa
                        in pipeline_spec["deploymentSpec"]["executors"]["exec-node1"][
                            "container"
                        ]["args"][0]
                    )
                    assert (
                        "--params test_param={{$.inputs.parameters['test_param']}},param2={{$.inputs.parameters['param2']}}"  # noqa
                        in pipeline_spec["deploymentSpec"]["executors"]["exec-node2"][
                            "container"
                        ]["args"][0]
                    )

    def mock_mlflow(self, enabled=False):
        def fakeimport(name, *args, **kw):
            if not enabled and (name == "mlflow" or name == "kedro_mlflow"):
                raise ImportError
            return self.realimport(name, *args, **kw)

        __builtins__["__import__"] = fakeimport

    def setUp(self):
        self.realimport = __builtins__["__import__"]
        self.mock_mlflow(False)

    def tearDown(self):
        __builtins__["__import__"] = self.realimport

    def create_generator(self, config={}, params={}, catalog={}):
        project_name = "my-awesome-project"
        config_loader = MagicMock()
        config_loader.get.return_value = catalog
        context = type(
            "obj",
            (object,),
            {
                "env": "unittests",
                "params": params,
                "config_loader": config_loader,
            },
        )

        self.pipelines_under_test = {"pipeline": self.create_pipeline()}

        config_with_defaults = {
            "image": "test",
            "root": "sample-bucket/sample-suffix",
            "experiment_name": "test-experiment",
            "run_name": "test-run",
        }
        config_with_defaults.update(config)
        self.generator_under_test = PipelineGenerator(
            PluginConfig.model_validate(
                {
                    "project_id": "test-project",
                    "region": "test-region",
                    "run_config": config_with_defaults,
                }
            ),
            project_name,
            context,
            "run-name",
        )
